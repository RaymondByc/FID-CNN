import tensorflow as tf
import collections
import glob
import os
import math
import time
from tensorflow.python import debug as tf_debug
import random
import numpy as np
from scipy import signal
from tensorflow.python.ops import math_ops


# 返回的CLASS的格式
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "grads_and_vars, loss, train, outputs")


BATCH_SIZE = 1
EPS = 1e-12
SUMMARY_FREQ = 100
TRACE_FREQ = 0
OUTDIR = 'out'
CHECKPOINT = 'out_success'
MAX_STEPS = None
MAX_EPOCH = 100 # number of training epochs
PROCESS_FREQ = 50 # display progress every progress_freq steps
DISPLAY_FREQ = 0  # write current training images every display_freq steps
SAVE_FREQ = 5000
INPUTDIR = 'data_train'
SCALE_SIZE = 286
FLIP = True
ASPECT_RATIO = 1.0
CROP_SIZE = 256
MODE = 'train'


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
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 1, message="image does not have 1 channel")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input) # 新的raw_input

        raw_input.set_shape([None, None, 1])

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
    loss = tf.reduce_mean(tf.square(targets - outputs)) + lambda_tv * frac_total_variation(outputs, 1.65, 2.75, 1.25)
    # loss = tf.reduce_mean(tf.square(targets - outputs)) + lambda_tv * frac_total_variation(outputs)
    # loss = tf.reduce_mean(tf.square(targets - outputs))

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
    for i, in_path in enumerate(fetches["paths:"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs:", "outputs:", "targets:"]:
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

        for kind in ["inputs:", "outputs:", "targets:"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

# 分数阶loss 可运行但是很慢版
# def fractional_variation(image, v1, v2, v3):
#
#     # preprocessed
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     image = image.eval(session=sess)
#     sess.close()
#
#     width = height = image.shape[1]
#
#     image = np.insert(image, 0, values=image[0,:], axis=0)
#     image = np.insert(image, width, values=image[width, :], axis=0)
#     image = np.insert(image, 0, values=image[:,0], axis=1)
#     image = np.insert(image, width, values=image[:,width], axis=1)
#
#
#     # 求差分过程中每一点的系数
#     def w_item(w_v, n):
#
#         if n == 0:
#             return 1
#
#         w_pro = 1 / math.factorial(n + 1)
#         for w_i in range(0, n + 1):
#             w_pro *= w_v + w_i
#         return w_pro
#
#     # x方向每个点对这个点的分数阶差分
#     def fx(x,y,image,v):
#         x_sum = 0
#
#         for dx in range(1, 17):
#             m = abs(dx - x)
#             x_sum += w_item(v, m) * image[dx, y]
#
#         return x_sum
#
#         # y方向上每一点对于该点的差分
#     def fy(x,y,image,v):
#         y_sum = 0
#
#         for dy in range(1, 17):
#             m = abs(dy - y)
#             y_sum += w_item(v, m) * image[x, dy]
#
#         return y_sum
#
#     # 存放差分数据
#     new_matric_x = np.ones((1, width, height, 1))
#     new_matric_y = np.ones((1, width, height, 1))
#     for x in range(1, image.shape[0] - 1):
#         for y in range(1, image.shape[1] - 1):
#             new_matric_x[0, x-1, y-1, 0] = fx(x, y, image, v1)
#             new_matric_y[0, x-1, y-1, 0] = fy(x, y, image, v1)
#
#     # 对已差分矩阵进行一阶差分
#     Ixx = (np.column_stack((new_matric_x[0, :, 1:, 0], new_matric_x[0, :, width-1, 0])) - np.column_stack((new_matric_x[0, :, 0, 0],new_matric_x[0, :, :width-1, 0])))/ 2
#     Iyy = (np.row_stack((new_matric_y[0, 1:, :, 0], new_matric_y[0, height-1, :, 0])) - np.row_stack((new_matric_y[0, 0, :, 0],new_matric_y[0, :height-1, :, 0]))) / 2
#
#     # 2norm 对x方向和y方向的进行平方再开方
#     new_matric_2norm = np.sqrt(np.power(new_matric_x, 2) + np.power(new_matric_y, 2))
#
#     # 堆叠方程
#     res = 0
#
#     for k in [0,1]:
#         pro = 1
#         for t in range(0, 2 * k + 1):
#             left_part = (g(2*k - v3) * np.power(new_matric_2norm, (v2 - 2*k))) / (g(-v3) * g(2*k - v3 + 1))
#             right_part = ((v2 - 2*k) * g(1 + v1) * g(2*k - v3 + 1)) / (2*k + 1) * g(v1) * g(-v3) * g(2*k - v3 + 2) * np.power(new_matric_2norm, v2-2*k-2) * (new_matric_x + new_matric_y)
#
#             pro *= (v2 - t + 1) / math.factorial(2*k) * np.sum(left_part + right_part)
#
#         res += pro
#
#     return tf.convert_to_tensor(res)

# def fractional_variation(image, v1, v2, v3):
#
#     g = math.gamma
#
#     # preprocessed
#
#     width = image.get_shape()[1]
#
#     image = tf.concat([image, image[:, (width - 1):, :, :]], 1)
#     image = tf.concat([image, image[:, :, (width - 1):, :]], 2)
#
#     # 求差分过程中每一点的系数
#     def w_item(w_v, n):
#
#         if n == 0:
#             return 1
#
#         w_pro = 1 / math.factorial(n + 1)
#         for w_i in range(0, n + 1):
#             w_pro *= w_v + w_i
#         return w_pro
#
#     # x方向每个点对这个点的分数阶差分
#     def fx(x,y,image,v):
#         x_sum = 0
#
#         for dx in range(1, width + 1):
#             m = abs(dx - x)
#             x_sum += w_item(v, m) * image[dx, y]
#
#         return x_sum
#
#         # y方向上每一点对于该点的差分
#     def fy(x,y,image,v):
#         y_sum = 0
#
#         for dy in range(1, width + 1):
#             m = abs(dy - y)
#             y_sum += w_item(v, m) * image[x, dy]
#
#         return y_sum
#
#     # 存放差分数据
#     new_matric_x = np.ones((width, width))
#     new_matric_y = np.ones((width, width))
#     for x in range(1, image.shape[0] - 1):
#         for y in range(1, image.shape[1] - 1):
#
#             new_matric_x[x-1, y-1] = fx(x, y, image, v1)
#             new_matric_y[x-1, y-1] = fy(x, y, image, v1)
#
#     # 对已差分矩阵进行一阶差分
#     Ixx = (np.column_stack((new_matric_x[:, 1:], new_matric_x[:, width - 1])) - np.column_stack((new_matric_x[:,0],new_matric_x[:, :width - 1])))/ 2
#     Iyy = (np.row_stack((new_matric_y[1:, :], new_matric_y[width - 1, :])) - np.row_stack((new_matric_y[0,:],new_matric_y[:width - 1, :]))) / 2
#
#     # 2norm 对x方向和y方向的进行平方再开方
#     new_matric_2norm = np.sqrt(np.power(new_matric_x, 2) + np.power(new_matric_y, 2))
#
#     # 堆叠方程
#     res = 0
#
#     for k in [0,1]:
#         pro = 1
#         for t in range(0, 2 * k + 1):
#             left_part = (g(2*k - v3) * np.power(new_matric_2norm, (v2 - 2*k))) / (g(-v3) * g(2*k - v3 + 1))
#             right_part = ((v2 - 2*k) * g(1 + v1) * g(2*k - v3 + 1)) / (2*k + 1) * g(v1) * g(-v3) * g(2*k - v3 + 2) * np.power(new_matric_2norm, v2-2*k-2) * (Ixx + Iyy)
#
#             pro *= (v2 - t + 1) / math.factorial(2*k) * np.sum(left_part + right_part)
#
#         res += pro
#     print(-res)
#     return -res

def frac_total_variation(images, v1=1.65, v2=2.75, v3=1.25, name=None):

  g = math.gamma

  with tf.name_scope(name, 'total_variation'):
    ndims = images.get_shape().ndims

    if ndims == 4:
      # generate the fractional array
      list = [(-v1)*(-v1+1)/2, -v1, 2, -v1, (-v1)*(-v1+1)/2]

      # generate the fractional filter
      filter_x = tf.constant(list, shape=(5, 1, 1, 1), dtype=tf.float32)
      filter_y = tf.constant(list, shape=(1, 5, 1, 1), dtype=tf.float32)

      # expand the tensor
      # mat_ex = tf.pad(images, [[0, 0], [2, 2], [0, 0], [0, 0]], "REFLECT")
      # mat_ey = tf.pad(images, [[0, 0], [0, 0], [2, 2], [0, 0]], "REFLECT")

      # def w_item(w_v, n):
      #
      #     if n == 0:
      #         return 1
      #
      #     w_pro = 1 / math.factorial(n + 1)
      #     for w_i in range(0, n + 1):
      #         w_pro *= w_v + w_i
      #     return w_pro
      #
      # def map_dx(ele, n):
      #     return ele * w_item(v1, n)

      # generate the dx_matrix
      dx_mat = tf.nn.conv2d(images, filter_x, [1, 1, 1, 1], 'SAME', True)
      dy_mat = tf.nn.conv2d(images, filter_y, [1, 1, 1, 1], 'SAME', True)

      pixel_dif1 = dx_mat[:, 1:, :, :] - dx_mat[:, :-1, :, :]
      pixel_dif2 = dy_mat[:, :, 1:, :] - dy_mat[:, :, :-1, :]

      # Only sum for the last 3 axis.
      # This results in a 1-D tensor with the total variation for each image.
      sum_axis = [1, 2, 3]
    else:
      raise ValueError('\'images\' must be either 3 or 4-dimensional.')

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_res = (math_ops.reduce_sum(math_ops.abs(pixel_dif1), axis=sum_axis) +
               math_ops.reduce_sum(math_ops.abs(pixel_dif2), axis=sum_axis))

    # 2norm 对x方向和y方向的进行平方再开方
    new_matric_2norm = tf.sqrt(tf.pow(dx_mat, 2) + tf.pow(dy_mat, 2))

    # 堆叠方程
    res = 0

    for k in [0,1]:
        pro = 1
        for t in range(0, 2 * k + 1):
            left_part = (g(2*k - v3) * tf.pow(new_matric_2norm, (v2 - 2*k))) / (g(-v3) * g(2*k - v3 + 1))
            right_part = ((v2 - 2*k) * g(1 + v1) * g(2*k - v3 + 1)) / (2*k + 1) * g(v1) * g(-v3) * g(2*k - v3 + 2) * np.power(new_matric_2norm, v2-2*k-2) * tot_res

            pro *= (v2 - t + 1) / math.factorial(2*k) * (left_part + right_part)

        res += pro


  return res

# load image data_train
# def fractional_variation(image, v1, v2, v3):
#
#     g = math.gamma
#     # preprocessed
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     image = image.eval(session=sess)
#     sess.close()
#
#     width = height = image.shape[1]
#
#     # image_x = np.insert(image, 0, values=image[0,:], axis=0)
#     # image_x = np.insert(image, width, values=image[width, :], axis=0)
#     # image_y = np.insert(image, 0, values=image[:,0], axis=1)
#     # image_y = np.insert(image, width, values=image[:,width], axis=1)
#
#
#     # # 求差分过程中每一点的系数
#     # def w_item(w_v, n):
#     #
#     #     if n == 0:
#     #         return 1
#     #
#     #     w_pro = 1 / math.factorial(n + 1)
#     #     for w_i in range(0, n + 1):
#     #         w_pro *= w_v + w_i
#     #     return w_pro
#     #
#     # # x方向每个点对这个点的分数阶差分
#     # def fx(x,y,image,v):
#     #     x_sum = 0
#     #
#     #     for dx in range(1, 17):
#     #         m = abs(dx - x)
#     #         x_sum += w_item(v, m) * image[dx, y]
#     #
#     #     return x_sum
#     #
#     #     # y方向上每一点对于该点的差分
#     # def fy(x,y,image,v):
#     #     y_sum = 0
#     #
#     #     for dy in range(1, 17):
#     #         m = abs(dy - y)
#     #         y_sum += w_item(v, m) * image[x, dy]
#     #
#     #     return y_sum
#     #
#     # # 存放差分数据
#     # new_matric_x = np.ones((1, width, height, 1))
#     # new_matric_y = np.ones((1, width, height, 1))
#     # for x in range(1, image.shape[0] - 1):
#     #     for y in range(1, image.shape[1] - 1):
#     #         new_matric_x[0, x-1, y-1, 0] = fx(x, y, image, v1)
#     #         new_matric_y[0, x-1, y-1, 0] = fy(x, y, image, v1)
#
#     # 对已差分矩阵进行一阶差分
#     list = [(-v1) * (-v1 + 1) / 2, -v1, 2, -v1, (-v1) * (-v1 + 1) / 2]
#     dx_filter = np.asarray(list).reshape((1,5,1,1))
#     dy_filter = np.asarray(list).reshape((1,1,5,1))
#
#     new_matric_x = tf.nn.conv2d(image, filter=dx_filter, mode='same')
#     new_matric_y = tf.nn.conv2d(image, filter=dy_filter, mode='same')
#
#     Ixx = (np.column_stack((new_matric_x[0, :, 1:, 0], new_matric_x[0, :, width-1, 0])) - np.column_stack((new_matric_x[0, :, 0, 0],new_matric_x[0, :, :width-1, 0])))/ 2
#     Iyy = (np.row_stack((new_matric_y[0, 1:, :, 0], new_matric_y[0, height-1, :, 0])) - np.row_stack((new_matric_y[0, 0, :, 0],new_matric_y[0, :height-1, :, 0]))) / 2
#
#     # 2norm 对x方向和y方向的进行平方再开方
#     new_matric_2norm = np.sqrt(np.power(new_matric_x, 2) + np.power(new_matric_y, 2))
#
#     # 堆叠方程
#     res = 0
#
#     for k in [0,1]:
#         pro = 1
#         for t in range(0, 2 * k + 1):
#             left_part = (g(2*k - v3) * np.power(new_matric_2norm, (v2 - 2*k))) / (g(-v3) * g(2*k - v3 + 1))
#             right_part = ((v2 - 2*k) * g(1 + v1) * g(2*k - v3 + 1)) / (2*k + 1) * g(v1) * g(-v3) * g(2*k - v3 + 2) * np.power(new_matric_2norm, v2-2*k-2) * (Ixx + Iyy)
#
#             pro *= (v2 - t + 1) / math.factorial(2*k) * np.sum(left_part + right_part)
#
#         res += pro
#
#     return tf.convert_to_tensor(res)
examples = load_examples()

# model return grads_and_vars, loss, train, outputs, step_update
model = create_model(examples.inputs, examples.targets)
print("examples count = %d" % examples.count)
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
        "paths:" : examples.paths,
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

#!! tf.summary.scalar("loss", model.loss)

with tf.name_scope("parameter_count"):

    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

saver = tf.train.Saver(max_to_keep=1)

log_dir = OUTDIR if (TRACE_FREQ > 0 or SUMMARY_FREQ > 0) else None
sv = tf.train.Supervisor(logdir=log_dir, save_summaries_secs=0, saver=None )
with sv.managed_session() as sess:
    print("parameter_count = ",sess.run(parameter_count))

    if CHECKPOINT is not None:
        checkpoint = tf.train.latest_checkpoint(CHECKPOINT)
        saver.restore(sess, checkpoint)

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

            results = sess.run(fetch, options=options, run_metadata=run_metadata)

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