import tensorflow as tf
import collections
import glob
import os
import math
import time
import random

# 返回的CLASS的格式
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "grads_and_vars, loss, train, outputs")


BATCH_SIZE = 1
EPS = 1e-12
SUMMARY_FREQ = 100
TRACE_FREQ = 0
OUTDIR = 'out'
CHECKPOINT = ''
MAX_STEPS = None
MAX_EPOCH = 100 # number of training epochs
PROCESS_FREQ = 50 # display progress every progress_freq steps
DISPLAY_FREQ = 0  # write current training images every display_freq steps
SAVE_FREQ = 5000
INPUTDIR = 'data'
SCALE_SIZE = 286
FLIP = True
ASPECT_RATIO = 1.0
CROP_SIZE = 256
MODE = 'test'


def preprocess(image): # ？这个预处理不知道是为什么
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2
def load_examples():
    if INPUTDIR is None or not os.path.exists(INPUTDIR):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(INPUTDIR, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(INPUTDIR, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue) # 返回了路径与内容队列，且返回的内容可以直接通过f,write写在文件中
        raw_input = decode(contents) # return A Tensor of type uint8. 3-D with shape [height, width, channels]

        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32) # 转化图片转化为float32类型

        # 验证图片必须为三维的
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input) # 新的raw_input

        raw_input.set_shape([None, None, 3])

        # lab 是个类似于 rgb的色彩空间

        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = preprocess(raw_input[:,:width//2,:])
        b_images = preprocess(raw_input[:,width//2:,:])


    inputs,targets = [a_images, b_images]

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [SCALE_SIZE, SCALE_SIZE], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, SCALE_SIZE - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if SCALE_SIZE > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif SCALE_SIZE < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=BATCH_SIZE)
    steps_per_epoch = int(math.ceil(len(input_paths) / BATCH_SIZE))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )
def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [3, 3, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        # padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="SAME")
        return conv
def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [3, 3, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height, in_width, out_channels], [1,1, 1, 1], padding="SAME")
        return conv
def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized
def create_model(inputs, targets):

    layers = []

    # layers = [e1,e2,e3,e4,d4,d3,d2,d1]
    # e - encode 编码
    # e1
    with tf.variable_scope("encode_1"):
        rect1 = tf.nn.leaky_relu(inputs, alpha=0.2)
        conv1 = conv(rect1,64,1)
        layers.append(conv1)

    # e2 - e4
    for e_i in range(2,5):
        with tf.variable_scope("encode_%d" % e_i):
            e_rected = tf.nn.leaky_relu(layers[-1], alpha=0.2)
            e_conved = conv(e_rected,64,1)
            e_normed = batchnorm(e_conved)
            layers.append(e_normed)

    with tf.variable_scope("decode_4"):
        rect_d4 = tf.nn.leaky_relu(layers[-1])
        conv_d4 = deconv(rect_d4, 64)
        norm_d4 = batchnorm(conv_d4)
        layers.append(norm_d4)

    # d3 - d2
    for d_i in [3,2]:
        with tf.variable_scope("decode_%d" % d_i):
            d_rected = tf.nn.relu(layers[-1])
            d_conved = deconv(d_rected,64)
            d_batched = batchnorm(d_conved)
            layers.append(d_batched)

    # d1
    with tf.variable_scope("decode_1"):
        d1_rect = tf.nn.relu(layers[-1])
        d1_conv = deconv(d1_rect, 1)
        layers.append(d1_conv)

    # 与初始图片相除
    o1_input = layers[-1]
    o1_rect = tf.nn.relu(o1_input)
    speckle_image = o1_rect + EPS

    outputs = tf.nn.tanh(inputs / speckle_image)

    # loss
    lambda_tv = 0.003 / (256 * 256)
    # loss = tf.reduce_mean(tf.square(targets - outputs)) + lambda_tv * tf.image.total_variation(outputs)
    loss = tf.reduce_mean(tf.square(targets - outputs))

    optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5) # 优化器
    grads_and_vars = optim.compute_gradients(loss) # 变量和梯度记录
    train = optim.apply_gradients(grads_and_vars) # trian

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    loss_update = ema.apply([loss])

    global_step = tf.train.get_or_create_global_step()
    step_update = tf.assign(global_step, global_step+1)

    return Model(outputs=outputs,
                 grads_and_vars=grads_and_vars,
                 train=tf.group(train, step_update,loss_update),
                 loss=ema.average(loss)
                 )
def convert(image): return  tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
def save_images(fetches, step=None):
    image_dir = os.path.join(OUTDIR, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets
# 添加到网页
def append_index(filesets, step=False):
    index_path = os.path.join(OUTDIR, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

if MODE == 'test':
    # options = {"which_direction", "ngf", "ndf", "lab_colorization"}
    # with open(os.path.join(CHECKPOINT, "options.json")) as f:
    #     for key, val in json.loads(f.read()).items():
    #         if key in options:
    #             print("loaded", key, "=", val)
                # setattr(a, key, val)
    # disable these features in test mode
    SCALE_SIZE = CROP_SIZE
    FLIP = False
# load image data
examples = load_examples()

# model return grads_and_vars, loss, train, outputs, step_update
model = create_model(examples.inputs, examples.targets)
# deprocess ???
inputs = deprocess(examples.inputs)
targets = deprocess(examples.targets)
outputs = deprocess(model.outputs)

with tf.name_scope("convert_inputs"):
    convert_inputs = convert(inputs)

with tf.name_scope("convert_outputs"):
    convert_outputs = convert(outputs)

with tf.name_scope("convert_targets"):
    convert_targets = convert(targets)

def ret_paths(path):
    return path

with tf.name_scope("encode_image"):
    display_fetch = {
        "paths:" : tf.map_fn(ret_paths, examples.paths, dtype=tf.string, name="paths"),
        "inputs:" : tf.map_fn(tf.image.encode_png, convert_inputs, dtype=tf.string, name="input_pngs"),
        "outputs:": tf.map_fn(tf.image.encode_png, convert_outputs, dtype=tf.string, name="output_pngs"),
        "targets:": tf.map_fn(tf.image.encode_png, convert_targets, dtype=tf.string, name="targets_pngs"),
    }

# summaries 打印数据
with tf.name_scope("inputs_summary"):
    tf.summary.image("inputs", convert_inputs)

with tf.name_scope("outputs_summary"):
    tf.summary.image("outputs", convert_outputs)

with tf.name_scope("targets_summary"):
    tf.summary.image("targets", convert_targets)

for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/values", var)

for grad, var in model.grads_and_vars:
    tf.summary.histogram(var.op.name + "/gradients", grad)

# tf.summary.scalar("loss", model.loss)

with tf.name_scope("parameter_count"):

    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

saver = tf.train.Saver(max_to_keep=1)

log_dir = OUTDIR if (TRACE_FREQ > 0 or SUMMARY_FREQ > 0) else None
sv = tf.train.Supervisor(logdir=log_dir, save_summaries_secs=0, saver=None )
with sv.managed_session() as sess:
    print("parameter_count = ",sess.run(parameter_count))

    if CHECKPOINT is not None:
        print("loading model.. and you need finish that")


    max_step = 2**32
    if MAX_EPOCH is not None:
        max_step = examples.steps_per_epoch * MAX_EPOCH
    if MAX_STEPS is not None:
        max_step = MAX_STEPS

    # test mode about max_step
    # train mode
    if MODE == 'train':
        start = time.time()

        for step in range(max_step):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_step - 1)

            options = None
            run_metadata = None

            if should(TRACE_FREQ):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetch = {
                "train": model.train,
                "global_step": sv.global_step,
                "loss": model.loss
            }

            if should(SUMMARY_FREQ):
                fetch["summary"] = sv.summary_op

            if should(DISPLAY_FREQ):
                fetch["display"] = display_fetch


            if should(SUMMARY_FREQ):
                print("recording summary")
                sv.summary_writer.add_summary(results["summary"], results["global_step"])

            if should(DISPLAY_FREQ):
                print("saving display images")
                filesets = save_images(results["display"], step=results["global_step"])
                append_index(filesets, step=True)

            if should(TRACE_FREQ):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

            if should(PROCESS_FREQ):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * BATCH_SIZE / (time.time() - start)
                remaining = (max_step - step) * BATCH_SIZE / rate
                print(
                    "progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                print("loss", results["loss"])

                if should(SAVE_FREQ):
                    print("saving model")
                    saver.save(sess, os.path.join(OUTDIR, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break
    elif MODE == 'test':
        max_step = min(examples.steps_per_epoch, max_step)


        for step in range(max_step):
            results = sess.run(display_fetch)
            filesets = save_images(results)
            for i, f in enumerate(filesets):
                print("evaluated image", f["name"])
            index_path = append_index(filesets)

            print("wrote index at", index_path)

