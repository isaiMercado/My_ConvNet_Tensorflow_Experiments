node {
  name: "targets_matrix"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "features_matrix"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "input_node/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\377\377\377\377@\000\000\000@\000\000\000\003\000\000\000"
      }
    }
  }
}
node {
  name: "input_node"
  op: "Reshape"
  input: "features_matrix"
  input: "input_node/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional1/truncated_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000\003\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "convolutional1/truncated_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional1/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "convolutional1/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "convolutional1/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "convolutional1/truncated_normal/mul"
  op: "Mul"
  input: "convolutional1/truncated_normal/TruncatedNormal"
  input: "convolutional1/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional1/truncated_normal"
  op: "Add"
  input: "convolutional1/truncated_normal/mul"
  input: "convolutional1/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional1/weight_matrix_conv1"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional1/weight_matrix_conv1/Assign"
  op: "Assign"
  input: "convolutional1/weight_matrix_conv1"
  input: "convolutional1/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional1/weight_matrix_conv1/read"
  op: "Identity"
  input: "convolutional1/weight_matrix_conv1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
}
node {
  name: "convolutional1/truncated_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 32
      }
    }
  }
}
node {
  name: "convolutional1/truncated_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional1/truncated_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "convolutional1/truncated_normal_1/TruncatedNormal"
  op: "TruncatedNormal"
  input: "convolutional1/truncated_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "convolutional1/truncated_normal_1/mul"
  op: "Mul"
  input: "convolutional1/truncated_normal_1/TruncatedNormal"
  input: "convolutional1/truncated_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional1/truncated_normal_1"
  op: "Add"
  input: "convolutional1/truncated_normal_1/mul"
  input: "convolutional1/truncated_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional1/bias_vector_conv1"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional1/bias_vector_conv1/Assign"
  op: "Assign"
  input: "convolutional1/bias_vector_conv1"
  input: "convolutional1/truncated_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional1/bias_vector_conv1/read"
  op: "Identity"
  input: "convolutional1/bias_vector_conv1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
}
node {
  name: "convolutional1/Conv2D"
  op: "Conv2D"
  input: "input_node"
  input: "convolutional1/weight_matrix_conv1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional1/add"
  op: "Add"
  input: "convolutional1/Conv2D"
  input: "convolutional1/bias_vector_conv1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional1/Relu"
  op: "Relu"
  input: "convolutional1/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional1/MaxPool"
  op: "MaxPool"
  input: "convolutional1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "convolutional2/truncated_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "convolutional2/truncated_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional2/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "convolutional2/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "convolutional2/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "convolutional2/truncated_normal/mul"
  op: "Mul"
  input: "convolutional2/truncated_normal/TruncatedNormal"
  input: "convolutional2/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional2/truncated_normal"
  op: "Add"
  input: "convolutional2/truncated_normal/mul"
  input: "convolutional2/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional2/weight_matrix_conv2"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional2/weight_matrix_conv2/Assign"
  op: "Assign"
  input: "convolutional2/weight_matrix_conv2"
  input: "convolutional2/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional2/weight_matrix_conv2/read"
  op: "Identity"
  input: "convolutional2/weight_matrix_conv2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
}
node {
  name: "convolutional2/truncated_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "convolutional2/truncated_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional2/truncated_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "convolutional2/truncated_normal_1/TruncatedNormal"
  op: "TruncatedNormal"
  input: "convolutional2/truncated_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "convolutional2/truncated_normal_1/mul"
  op: "Mul"
  input: "convolutional2/truncated_normal_1/TruncatedNormal"
  input: "convolutional2/truncated_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional2/truncated_normal_1"
  op: "Add"
  input: "convolutional2/truncated_normal_1/mul"
  input: "convolutional2/truncated_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional2/bias_vector_conv2"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional2/bias_vector_conv2/Assign"
  op: "Assign"
  input: "convolutional2/bias_vector_conv2"
  input: "convolutional2/truncated_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional2/bias_vector_conv2/read"
  op: "Identity"
  input: "convolutional2/bias_vector_conv2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
}
node {
  name: "convolutional2/Conv2D"
  op: "Conv2D"
  input: "convolutional1/MaxPool"
  input: "convolutional2/weight_matrix_conv2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional2/add"
  op: "Add"
  input: "convolutional2/Conv2D"
  input: "convolutional2/bias_vector_conv2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional2/Relu"
  op: "Relu"
  input: "convolutional2/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "convolutional2/MaxPool"
  op: "MaxPool"
  input: "convolutional2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "convolutional2/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\000@\000\000"
      }
    }
  }
}
node {
  name: "convolutional2/Reshape"
  op: "Reshape"
  input: "convolutional2/MaxPool"
  input: "convolutional2/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000@\000\000\000\010\000\000"
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "fully_connected_hidden1/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal/mul"
  op: "Mul"
  input: "fully_connected_hidden1/truncated_normal/TruncatedNormal"
  input: "fully_connected_hidden1/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal"
  op: "Add"
  input: "fully_connected_hidden1/truncated_normal/mul"
  input: "fully_connected_hidden1/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_hidden1/weight_matrix_fc1"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16384
        }
        dim {
          size: 2048
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_hidden1/weight_matrix_fc1/Assign"
  op: "Assign"
  input: "fully_connected_hidden1/weight_matrix_fc1"
  input: "fully_connected_hidden1/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_hidden1/weight_matrix_fc1/read"
  op: "Identity"
  input: "fully_connected_hidden1/weight_matrix_fc1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2048
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal_1/TruncatedNormal"
  op: "TruncatedNormal"
  input: "fully_connected_hidden1/truncated_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal_1/mul"
  op: "Mul"
  input: "fully_connected_hidden1/truncated_normal_1/TruncatedNormal"
  input: "fully_connected_hidden1/truncated_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_hidden1/truncated_normal_1"
  op: "Add"
  input: "fully_connected_hidden1/truncated_normal_1/mul"
  input: "fully_connected_hidden1/truncated_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_hidden1/bias_vector_fc1"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2048
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_hidden1/bias_vector_fc1/Assign"
  op: "Assign"
  input: "fully_connected_hidden1/bias_vector_fc1"
  input: "fully_connected_hidden1/truncated_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_hidden1/bias_vector_fc1/read"
  op: "Identity"
  input: "fully_connected_hidden1/bias_vector_fc1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/MatMul"
  op: "MatMul"
  input: "convolutional2/Reshape"
  input: "fully_connected_hidden1/weight_matrix_fc1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "fully_connected_hidden1/add"
  op: "Add"
  input: "fully_connected_hidden1/MatMul"
  input: "fully_connected_hidden1/bias_vector_fc1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_hidden1/Relu"
  op: "Relu"
  input: "fully_connected_hidden1/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\010\000\000\004\000\000\000"
      }
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "fully_connected_output/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal/mul"
  op: "Mul"
  input: "fully_connected_output/truncated_normal/TruncatedNormal"
  input: "fully_connected_output/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal"
  op: "Add"
  input: "fully_connected_output/truncated_normal/mul"
  input: "fully_connected_output/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_output/weight_matrix_fc2"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2048
        }
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_output/weight_matrix_fc2/Assign"
  op: "Assign"
  input: "fully_connected_output/weight_matrix_fc2"
  input: "fully_connected_output/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_output/weight_matrix_fc2/read"
  op: "Identity"
  input: "fully_connected_output/weight_matrix_fc2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 4
      }
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal_1/TruncatedNormal"
  op: "TruncatedNormal"
  input: "fully_connected_output/truncated_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal_1/mul"
  op: "Mul"
  input: "fully_connected_output/truncated_normal_1/TruncatedNormal"
  input: "fully_connected_output/truncated_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_output/truncated_normal_1"
  op: "Add"
  input: "fully_connected_output/truncated_normal_1/mul"
  input: "fully_connected_output/truncated_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_output/bias_vector_fc2"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_output/bias_vector_fc2/Assign"
  op: "Assign"
  input: "fully_connected_output/bias_vector_fc2"
  input: "fully_connected_output/truncated_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_output/bias_vector_fc2/read"
  op: "Identity"
  input: "fully_connected_output/bias_vector_fc2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
}
node {
  name: "fully_connected_output/MatMul"
  op: "MatMul"
  input: "fully_connected_hidden1/Relu"
  input: "fully_connected_output/weight_matrix_fc2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "fully_connected_output/add"
  op: "Add"
  input: "fully_connected_output/MatMul"
  input: "fully_connected_output/bias_vector_fc2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "fully_connected_output/output_node"
  op: "Softmax"
  input: "fully_connected_output/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Log"
  op: "Log"
  input: "fully_connected_output/output_node"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "mul"
  op: "Mul"
  input: "targets_matrix"
  input: "Log"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Sum/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Sum"
  op: "Sum"
  input: "mul"
  input: "Sum/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Neg"
  op: "Neg"
  input: "Sum"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Mean"
  op: "Mean"
  input: "Neg"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/Shape"
  op: "Shape"
  input: "Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "gradients/Fill"
  op: "Fill"
  input: "gradients/Shape"
  input: "gradients/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Mean_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Reshape"
  op: "Reshape"
  input: "gradients/Fill"
  input: "gradients/Mean_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Mean_grad/Shape"
  op: "Shape"
  input: "Neg"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Mean_grad/Tile"
  op: "Tile"
  input: "gradients/Mean_grad/Reshape"
  input: "gradients/Mean_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Mean_grad/Shape_1"
  op: "Shape"
  input: "Neg"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Mean_grad/Shape_2"
  op: "Shape"
  input: "Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Mean_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Prod"
  op: "Prod"
  input: "gradients/Mean_grad/Shape_1"
  input: "gradients/Mean_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/Mean_grad/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Prod_1"
  op: "Prod"
  input: "gradients/Mean_grad/Shape_2"
  input: "gradients/Mean_grad/Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/Mean_grad/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Maximum"
  op: "Maximum"
  input: "gradients/Mean_grad/Prod_1"
  input: "gradients/Mean_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/floordiv"
  op: "Div"
  input: "gradients/Mean_grad/Prod"
  input: "gradients/Mean_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/Cast"
  op: "Cast"
  input: "gradients/Mean_grad/floordiv"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/truediv"
  op: "Div"
  input: "gradients/Mean_grad/Tile"
  input: "gradients/Mean_grad/Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Neg_grad/Neg"
  op: "Neg"
  input: "gradients/Mean_grad/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Sum_grad/Shape"
  op: "Shape"
  input: "mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Sum_grad/Size"
  op: "Size"
  input: "gradients/Sum_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Sum_grad/add"
  op: "Add"
  input: "Sum/reduction_indices"
  input: "gradients/Sum_grad/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Sum_grad/mod"
  op: "Mod"
  input: "gradients/Sum_grad/add"
  input: "gradients/Sum_grad/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Sum_grad/Shape_1"
  op: "Shape"
  input: "gradients/Sum_grad/mod"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Sum_grad/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients/Sum_grad/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/Sum_grad/range"
  op: "Range"
  input: "gradients/Sum_grad/range/start"
  input: "gradients/Sum_grad/Size"
  input: "gradients/Sum_grad/range/delta"
}
node {
  name: "gradients/Sum_grad/Fill/value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/Sum_grad/Fill"
  op: "Fill"
  input: "gradients/Sum_grad/Shape_1"
  input: "gradients/Sum_grad/Fill/value"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Sum_grad/DynamicStitch"
  op: "DynamicStitch"
  input: "gradients/Sum_grad/range"
  input: "gradients/Sum_grad/mod"
  input: "gradients/Sum_grad/Shape"
  input: "gradients/Sum_grad/Fill"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Sum_grad/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/Sum_grad/Maximum"
  op: "Maximum"
  input: "gradients/Sum_grad/DynamicStitch"
  input: "gradients/Sum_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Sum_grad/floordiv"
  op: "Div"
  input: "gradients/Sum_grad/Shape"
  input: "gradients/Sum_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Sum_grad/Reshape"
  op: "Reshape"
  input: "gradients/Neg_grad/Neg"
  input: "gradients/Sum_grad/DynamicStitch"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Sum_grad/Tile"
  op: "Tile"
  input: "gradients/Sum_grad/Reshape"
  input: "gradients/Sum_grad/floordiv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/mul_grad/Shape"
  op: "Shape"
  input: "targets_matrix"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/mul_grad/Shape_1"
  op: "Shape"
  input: "Log"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/mul_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/mul_grad/Shape"
  input: "gradients/mul_grad/Shape_1"
}
node {
  name: "gradients/mul_grad/mul"
  op: "Mul"
  input: "gradients/Sum_grad/Tile"
  input: "Log"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/mul_grad/Sum"
  op: "Sum"
  input: "gradients/mul_grad/mul"
  input: "gradients/mul_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/mul_grad/Reshape"
  op: "Reshape"
  input: "gradients/mul_grad/Sum"
  input: "gradients/mul_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/mul_grad/mul_1"
  op: "Mul"
  input: "targets_matrix"
  input: "gradients/Sum_grad/Tile"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/mul_grad/Sum_1"
  op: "Sum"
  input: "gradients/mul_grad/mul_1"
  input: "gradients/mul_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/mul_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/mul_grad/Sum_1"
  input: "gradients/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/mul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/mul_grad/Reshape"
  input: "^gradients/mul_grad/Reshape_1"
}
node {
  name: "gradients/mul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/mul_grad/Reshape"
  input: "^gradients/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/mul_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/mul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/mul_grad/Reshape_1"
  input: "^gradients/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/mul_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/Log_grad/Inv"
  op: "Inv"
  input: "fully_connected_output/output_node"
  input: "^gradients/mul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Log_grad/mul"
  op: "Mul"
  input: "gradients/mul_grad/tuple/control_dependency_1"
  input: "gradients/Log_grad/Inv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_output/output_node_grad/mul"
  op: "Mul"
  input: "gradients/Log_grad/mul"
  input: "fully_connected_output/output_node"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_output/output_node_grad/Sum/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/fully_connected_output/output_node_grad/Sum"
  op: "Sum"
  input: "gradients/fully_connected_output/output_node_grad/mul"
  input: "gradients/fully_connected_output/output_node_grad/Sum/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/fully_connected_output/output_node_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "gradients/fully_connected_output/output_node_grad/Reshape"
  op: "Reshape"
  input: "gradients/fully_connected_output/output_node_grad/Sum"
  input: "gradients/fully_connected_output/output_node_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_output/output_node_grad/sub"
  op: "Sub"
  input: "gradients/Log_grad/mul"
  input: "gradients/fully_connected_output/output_node_grad/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_output/output_node_grad/mul_1"
  op: "Mul"
  input: "gradients/fully_connected_output/output_node_grad/sub"
  input: "fully_connected_output/output_node"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_output/add_grad/Shape"
  op: "Shape"
  input: "fully_connected_output/MatMul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_output/add_grad/Shape_1"
  op: "Shape"
  input: "fully_connected_output/bias_vector_fc2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_output/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/fully_connected_output/add_grad/Shape"
  input: "gradients/fully_connected_output/add_grad/Shape_1"
}
node {
  name: "gradients/fully_connected_output/add_grad/Sum"
  op: "Sum"
  input: "gradients/fully_connected_output/output_node_grad/mul_1"
  input: "gradients/fully_connected_output/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/fully_connected_output/add_grad/Reshape"
  op: "Reshape"
  input: "gradients/fully_connected_output/add_grad/Sum"
  input: "gradients/fully_connected_output/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_output/add_grad/Sum_1"
  op: "Sum"
  input: "gradients/fully_connected_output/output_node_grad/mul_1"
  input: "gradients/fully_connected_output/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/fully_connected_output/add_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/fully_connected_output/add_grad/Sum_1"
  input: "gradients/fully_connected_output/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_output/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/fully_connected_output/add_grad/Reshape"
  input: "^gradients/fully_connected_output/add_grad/Reshape_1"
}
node {
  name: "gradients/fully_connected_output/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/fully_connected_output/add_grad/Reshape"
  input: "^gradients/fully_connected_output/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/fully_connected_output/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/fully_connected_output/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/fully_connected_output/add_grad/Reshape_1"
  input: "^gradients/fully_connected_output/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/fully_connected_output/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/fully_connected_output/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients/fully_connected_output/add_grad/tuple/control_dependency"
  input: "fully_connected_output/weight_matrix_fc2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/fully_connected_output/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "fully_connected_hidden1/Relu"
  input: "gradients/fully_connected_output/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/fully_connected_output/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/fully_connected_output/MatMul_grad/MatMul"
  input: "^gradients/fully_connected_output/MatMul_grad/MatMul_1"
}
node {
  name: "gradients/fully_connected_output/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/fully_connected_output/MatMul_grad/MatMul"
  input: "^gradients/fully_connected_output/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/fully_connected_output/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients/fully_connected_output/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/fully_connected_output/MatMul_grad/MatMul_1"
  input: "^gradients/fully_connected_output/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/fully_connected_output/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/fully_connected_output/MatMul_grad/tuple/control_dependency"
  input: "fully_connected_hidden1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/add_grad/Shape"
  op: "Shape"
  input: "fully_connected_hidden1/MatMul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/add_grad/Shape_1"
  op: "Shape"
  input: "fully_connected_hidden1/bias_vector_fc1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/fully_connected_hidden1/add_grad/Shape"
  input: "gradients/fully_connected_hidden1/add_grad/Shape_1"
}
node {
  name: "gradients/fully_connected_hidden1/add_grad/Sum"
  op: "Sum"
  input: "gradients/fully_connected_hidden1/Relu_grad/ReluGrad"
  input: "gradients/fully_connected_hidden1/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/add_grad/Reshape"
  op: "Reshape"
  input: "gradients/fully_connected_hidden1/add_grad/Sum"
  input: "gradients/fully_connected_hidden1/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/add_grad/Sum_1"
  op: "Sum"
  input: "gradients/fully_connected_hidden1/Relu_grad/ReluGrad"
  input: "gradients/fully_connected_hidden1/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/add_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/fully_connected_hidden1/add_grad/Sum_1"
  input: "gradients/fully_connected_hidden1/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/fully_connected_hidden1/add_grad/Reshape"
  input: "^gradients/fully_connected_hidden1/add_grad/Reshape_1"
}
node {
  name: "gradients/fully_connected_hidden1/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/fully_connected_hidden1/add_grad/Reshape"
  input: "^gradients/fully_connected_hidden1/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/fully_connected_hidden1/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/fully_connected_hidden1/add_grad/Reshape_1"
  input: "^gradients/fully_connected_hidden1/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/fully_connected_hidden1/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients/fully_connected_hidden1/add_grad/tuple/control_dependency"
  input: "fully_connected_hidden1/weight_matrix_fc1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "convolutional2/Reshape"
  input: "gradients/fully_connected_hidden1/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/fully_connected_hidden1/MatMul_grad/MatMul"
  input: "^gradients/fully_connected_hidden1/MatMul_grad/MatMul_1"
}
node {
  name: "gradients/fully_connected_hidden1/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/fully_connected_hidden1/MatMul_grad/MatMul"
  input: "^gradients/fully_connected_hidden1/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/fully_connected_hidden1/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients/fully_connected_hidden1/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/fully_connected_hidden1/MatMul_grad/MatMul_1"
  input: "^gradients/fully_connected_hidden1/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/fully_connected_hidden1/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "gradients/convolutional2/Reshape_grad/Shape"
  op: "Shape"
  input: "convolutional2/MaxPool"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional2/Reshape_grad/Reshape"
  op: "Reshape"
  input: "gradients/fully_connected_hidden1/MatMul_grad/tuple/control_dependency"
  input: "gradients/convolutional2/Reshape_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional2/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "convolutional2/Relu"
  input: "convolutional2/MaxPool"
  input: "gradients/convolutional2/Reshape_grad/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "gradients/convolutional2/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/convolutional2/MaxPool_grad/MaxPoolGrad"
  input: "convolutional2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional2/add_grad/Shape"
  op: "Shape"
  input: "convolutional2/Conv2D"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional2/add_grad/Shape_1"
  op: "Shape"
  input: "convolutional2/bias_vector_conv2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional2/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/convolutional2/add_grad/Shape"
  input: "gradients/convolutional2/add_grad/Shape_1"
}
node {
  name: "gradients/convolutional2/add_grad/Sum"
  op: "Sum"
  input: "gradients/convolutional2/Relu_grad/ReluGrad"
  input: "gradients/convolutional2/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/convolutional2/add_grad/Reshape"
  op: "Reshape"
  input: "gradients/convolutional2/add_grad/Sum"
  input: "gradients/convolutional2/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional2/add_grad/Sum_1"
  op: "Sum"
  input: "gradients/convolutional2/Relu_grad/ReluGrad"
  input: "gradients/convolutional2/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/convolutional2/add_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/convolutional2/add_grad/Sum_1"
  input: "gradients/convolutional2/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional2/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/convolutional2/add_grad/Reshape"
  input: "^gradients/convolutional2/add_grad/Reshape_1"
}
node {
  name: "gradients/convolutional2/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/convolutional2/add_grad/Reshape"
  input: "^gradients/convolutional2/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/convolutional2/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/convolutional2/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/convolutional2/add_grad/Reshape_1"
  input: "^gradients/convolutional2/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/convolutional2/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/convolutional2/Conv2D_grad/Shape"
  op: "Shape"
  input: "convolutional1/MaxPool"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional2/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "gradients/convolutional2/Conv2D_grad/Shape"
  input: "convolutional2/weight_matrix_conv2/read"
  input: "gradients/convolutional2/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/convolutional2/Conv2D_grad/Shape_1"
  op: "Shape"
  input: "convolutional2/weight_matrix_conv2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional2/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "convolutional1/MaxPool"
  input: "gradients/convolutional2/Conv2D_grad/Shape_1"
  input: "gradients/convolutional2/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/convolutional2/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/convolutional2/Conv2D_grad/Conv2DBackpropInput"
  input: "^gradients/convolutional2/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "gradients/convolutional2/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/convolutional2/Conv2D_grad/Conv2DBackpropInput"
  input: "^gradients/convolutional2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/convolutional2/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "gradients/convolutional2/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/convolutional2/Conv2D_grad/Conv2DBackpropFilter"
  input: "^gradients/convolutional2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/convolutional2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "gradients/convolutional1/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "convolutional1/Relu"
  input: "convolutional1/MaxPool"
  input: "gradients/convolutional2/Conv2D_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "gradients/convolutional1/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/convolutional1/MaxPool_grad/MaxPoolGrad"
  input: "convolutional1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional1/add_grad/Shape"
  op: "Shape"
  input: "convolutional1/Conv2D"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional1/add_grad/Shape_1"
  op: "Shape"
  input: "convolutional1/bias_vector_conv1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional1/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/convolutional1/add_grad/Shape"
  input: "gradients/convolutional1/add_grad/Shape_1"
}
node {
  name: "gradients/convolutional1/add_grad/Sum"
  op: "Sum"
  input: "gradients/convolutional1/Relu_grad/ReluGrad"
  input: "gradients/convolutional1/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/convolutional1/add_grad/Reshape"
  op: "Reshape"
  input: "gradients/convolutional1/add_grad/Sum"
  input: "gradients/convolutional1/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional1/add_grad/Sum_1"
  op: "Sum"
  input: "gradients/convolutional1/Relu_grad/ReluGrad"
  input: "gradients/convolutional1/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/convolutional1/add_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/convolutional1/add_grad/Sum_1"
  input: "gradients/convolutional1/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional1/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/convolutional1/add_grad/Reshape"
  input: "^gradients/convolutional1/add_grad/Reshape_1"
}
node {
  name: "gradients/convolutional1/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/convolutional1/add_grad/Reshape"
  input: "^gradients/convolutional1/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/convolutional1/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/convolutional1/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/convolutional1/add_grad/Reshape_1"
  input: "^gradients/convolutional1/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/convolutional1/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/convolutional1/Conv2D_grad/Shape"
  op: "Shape"
  input: "input_node"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional1/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "gradients/convolutional1/Conv2D_grad/Shape"
  input: "convolutional1/weight_matrix_conv1/read"
  input: "gradients/convolutional1/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/convolutional1/Conv2D_grad/Shape_1"
  op: "Shape"
  input: "convolutional1/weight_matrix_conv1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/convolutional1/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "input_node"
  input: "gradients/convolutional1/Conv2D_grad/Shape_1"
  input: "gradients/convolutional1/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/convolutional1/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/convolutional1/Conv2D_grad/Conv2DBackpropInput"
  input: "^gradients/convolutional1/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "gradients/convolutional1/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/convolutional1/Conv2D_grad/Conv2DBackpropInput"
  input: "^gradients/convolutional1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/convolutional1/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "gradients/convolutional1/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/convolutional1/Conv2D_grad/Conv2DBackpropFilter"
  input: "^gradients/convolutional1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/convolutional1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "beta1_power/initial_value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.899999976158
      }
    }
  }
}
node {
  name: "beta1_power"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "beta1_power/Assign"
  op: "Assign"
  input: "beta1_power"
  input: "beta1_power/initial_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "beta1_power/read"
  op: "Identity"
  input: "beta1_power"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
}
node {
  name: "beta2_power/initial_value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.999000012875
      }
    }
  }
}
node {
  name: "beta2_power"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "beta2_power/Assign"
  op: "Assign"
  input: "beta2_power"
  input: "beta2_power/initial_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "beta2_power/read"
  op: "Identity"
  input: "beta2_power"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
}
node {
  name: "zeros"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 5
          }
          dim {
            size: 5
          }
          dim {
            size: 3
          }
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional1/weight_matrix_conv1/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional1/weight_matrix_conv1/Adam/Assign"
  op: "Assign"
  input: "convolutional1/weight_matrix_conv1/Adam"
  input: "zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional1/weight_matrix_conv1/Adam/read"
  op: "Identity"
  input: "convolutional1/weight_matrix_conv1/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
}
node {
  name: "zeros_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 5
          }
          dim {
            size: 5
          }
          dim {
            size: 3
          }
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional1/weight_matrix_conv1/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional1/weight_matrix_conv1/Adam_1/Assign"
  op: "Assign"
  input: "convolutional1/weight_matrix_conv1/Adam_1"
  input: "zeros_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional1/weight_matrix_conv1/Adam_1/read"
  op: "Identity"
  input: "convolutional1/weight_matrix_conv1/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
}
node {
  name: "zeros_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional1/bias_vector_conv1/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional1/bias_vector_conv1/Adam/Assign"
  op: "Assign"
  input: "convolutional1/bias_vector_conv1/Adam"
  input: "zeros_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional1/bias_vector_conv1/Adam/read"
  op: "Identity"
  input: "convolutional1/bias_vector_conv1/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
}
node {
  name: "zeros_3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional1/bias_vector_conv1/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional1/bias_vector_conv1/Adam_1/Assign"
  op: "Assign"
  input: "convolutional1/bias_vector_conv1/Adam_1"
  input: "zeros_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional1/bias_vector_conv1/Adam_1/read"
  op: "Identity"
  input: "convolutional1/bias_vector_conv1/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
}
node {
  name: "zeros_4"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 5
          }
          dim {
            size: 5
          }
          dim {
            size: 32
          }
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional2/weight_matrix_conv2/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional2/weight_matrix_conv2/Adam/Assign"
  op: "Assign"
  input: "convolutional2/weight_matrix_conv2/Adam"
  input: "zeros_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional2/weight_matrix_conv2/Adam/read"
  op: "Identity"
  input: "convolutional2/weight_matrix_conv2/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
}
node {
  name: "zeros_5"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 5
          }
          dim {
            size: 5
          }
          dim {
            size: 32
          }
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional2/weight_matrix_conv2/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional2/weight_matrix_conv2/Adam_1/Assign"
  op: "Assign"
  input: "convolutional2/weight_matrix_conv2/Adam_1"
  input: "zeros_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional2/weight_matrix_conv2/Adam_1/read"
  op: "Identity"
  input: "convolutional2/weight_matrix_conv2/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
}
node {
  name: "zeros_6"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional2/bias_vector_conv2/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional2/bias_vector_conv2/Adam/Assign"
  op: "Assign"
  input: "convolutional2/bias_vector_conv2/Adam"
  input: "zeros_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional2/bias_vector_conv2/Adam/read"
  op: "Identity"
  input: "convolutional2/bias_vector_conv2/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
}
node {
  name: "zeros_7"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "convolutional2/bias_vector_conv2/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "convolutional2/bias_vector_conv2/Adam_1/Assign"
  op: "Assign"
  input: "convolutional2/bias_vector_conv2/Adam_1"
  input: "zeros_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "convolutional2/bias_vector_conv2/Adam_1/read"
  op: "Identity"
  input: "convolutional2/bias_vector_conv2/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
}
node {
  name: "zeros_8"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16384
          }
          dim {
            size: 2048
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/weight_matrix_fc1/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16384
        }
        dim {
          size: 2048
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_hidden1/weight_matrix_fc1/Adam/Assign"
  op: "Assign"
  input: "fully_connected_hidden1/weight_matrix_fc1/Adam"
  input: "zeros_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_hidden1/weight_matrix_fc1/Adam/read"
  op: "Identity"
  input: "fully_connected_hidden1/weight_matrix_fc1/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
}
node {
  name: "zeros_9"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16384
          }
          dim {
            size: 2048
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/weight_matrix_fc1/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16384
        }
        dim {
          size: 2048
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_hidden1/weight_matrix_fc1/Adam_1/Assign"
  op: "Assign"
  input: "fully_connected_hidden1/weight_matrix_fc1/Adam_1"
  input: "zeros_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_hidden1/weight_matrix_fc1/Adam_1/read"
  op: "Identity"
  input: "fully_connected_hidden1/weight_matrix_fc1/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
}
node {
  name: "zeros_10"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2048
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/bias_vector_fc1/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2048
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_hidden1/bias_vector_fc1/Adam/Assign"
  op: "Assign"
  input: "fully_connected_hidden1/bias_vector_fc1/Adam"
  input: "zeros_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_hidden1/bias_vector_fc1/Adam/read"
  op: "Identity"
  input: "fully_connected_hidden1/bias_vector_fc1/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
}
node {
  name: "zeros_11"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2048
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_hidden1/bias_vector_fc1/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2048
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_hidden1/bias_vector_fc1/Adam_1/Assign"
  op: "Assign"
  input: "fully_connected_hidden1/bias_vector_fc1/Adam_1"
  input: "zeros_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_hidden1/bias_vector_fc1/Adam_1/read"
  op: "Identity"
  input: "fully_connected_hidden1/bias_vector_fc1/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
}
node {
  name: "zeros_12"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2048
          }
          dim {
            size: 4
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_output/weight_matrix_fc2/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2048
        }
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_output/weight_matrix_fc2/Adam/Assign"
  op: "Assign"
  input: "fully_connected_output/weight_matrix_fc2/Adam"
  input: "zeros_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_output/weight_matrix_fc2/Adam/read"
  op: "Identity"
  input: "fully_connected_output/weight_matrix_fc2/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
}
node {
  name: "zeros_13"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2048
          }
          dim {
            size: 4
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_output/weight_matrix_fc2/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2048
        }
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_output/weight_matrix_fc2/Adam_1/Assign"
  op: "Assign"
  input: "fully_connected_output/weight_matrix_fc2/Adam_1"
  input: "zeros_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_output/weight_matrix_fc2/Adam_1/read"
  op: "Identity"
  input: "fully_connected_output/weight_matrix_fc2/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
}
node {
  name: "zeros_14"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_output/bias_vector_fc2/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_output/bias_vector_fc2/Adam/Assign"
  op: "Assign"
  input: "fully_connected_output/bias_vector_fc2/Adam"
  input: "zeros_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_output/bias_vector_fc2/Adam/read"
  op: "Identity"
  input: "fully_connected_output/bias_vector_fc2/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
}
node {
  name: "zeros_15"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "fully_connected_output/bias_vector_fc2/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "fully_connected_output/bias_vector_fc2/Adam_1/Assign"
  op: "Assign"
  input: "fully_connected_output/bias_vector_fc2/Adam_1"
  input: "zeros_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "fully_connected_output/bias_vector_fc2/Adam_1/read"
  op: "Identity"
  input: "fully_connected_output/bias_vector_fc2/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
}
node {
  name: "Adam/learning_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.99999974738e-05
      }
    }
  }
}
node {
  name: "Adam/beta1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.899999976158
      }
    }
  }
}
node {
  name: "Adam/beta2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.999000012875
      }
    }
  }
}
node {
  name: "Adam/epsilon"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.99999993923e-09
      }
    }
  }
}
node {
  name: "Adam/update_convolutional1/weight_matrix_conv1/ApplyAdam"
  op: "ApplyAdam"
  input: "convolutional1/weight_matrix_conv1"
  input: "convolutional1/weight_matrix_conv1/Adam"
  input: "convolutional1/weight_matrix_conv1/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/convolutional1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_convolutional1/bias_vector_conv1/ApplyAdam"
  op: "ApplyAdam"
  input: "convolutional1/bias_vector_conv1"
  input: "convolutional1/bias_vector_conv1/Adam"
  input: "convolutional1/bias_vector_conv1/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/convolutional1/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_convolutional2/weight_matrix_conv2/ApplyAdam"
  op: "ApplyAdam"
  input: "convolutional2/weight_matrix_conv2"
  input: "convolutional2/weight_matrix_conv2/Adam"
  input: "convolutional2/weight_matrix_conv2/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/convolutional2/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_convolutional2/bias_vector_conv2/ApplyAdam"
  op: "ApplyAdam"
  input: "convolutional2/bias_vector_conv2"
  input: "convolutional2/bias_vector_conv2/Adam"
  input: "convolutional2/bias_vector_conv2/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/convolutional2/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_fully_connected_hidden1/weight_matrix_fc1/ApplyAdam"
  op: "ApplyAdam"
  input: "fully_connected_hidden1/weight_matrix_fc1"
  input: "fully_connected_hidden1/weight_matrix_fc1/Adam"
  input: "fully_connected_hidden1/weight_matrix_fc1/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/fully_connected_hidden1/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_fully_connected_hidden1/bias_vector_fc1/ApplyAdam"
  op: "ApplyAdam"
  input: "fully_connected_hidden1/bias_vector_fc1"
  input: "fully_connected_hidden1/bias_vector_fc1/Adam"
  input: "fully_connected_hidden1/bias_vector_fc1/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/fully_connected_hidden1/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_fully_connected_output/weight_matrix_fc2/ApplyAdam"
  op: "ApplyAdam"
  input: "fully_connected_output/weight_matrix_fc2"
  input: "fully_connected_output/weight_matrix_fc2/Adam"
  input: "fully_connected_output/weight_matrix_fc2/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/fully_connected_output/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_fully_connected_output/bias_vector_fc2/ApplyAdam"
  op: "ApplyAdam"
  input: "fully_connected_output/bias_vector_fc2"
  input: "fully_connected_output/bias_vector_fc2/Adam"
  input: "fully_connected_output/bias_vector_fc2/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/fully_connected_output/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/mul"
  op: "Mul"
  input: "beta1_power/read"
  input: "Adam/beta1"
  input: "^Adam/update_convolutional1/weight_matrix_conv1/ApplyAdam"
  input: "^Adam/update_convolutional1/bias_vector_conv1/ApplyAdam"
  input: "^Adam/update_convolutional2/weight_matrix_conv2/ApplyAdam"
  input: "^Adam/update_convolutional2/bias_vector_conv2/ApplyAdam"
  input: "^Adam/update_fully_connected_hidden1/weight_matrix_fc1/ApplyAdam"
  input: "^Adam/update_fully_connected_hidden1/bias_vector_fc1/ApplyAdam"
  input: "^Adam/update_fully_connected_output/weight_matrix_fc2/ApplyAdam"
  input: "^Adam/update_fully_connected_output/bias_vector_fc2/ApplyAdam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
}
node {
  name: "Adam/Assign"
  op: "Assign"
  input: "beta1_power"
  input: "Adam/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Adam/mul_1"
  op: "Mul"
  input: "beta2_power/read"
  input: "Adam/beta2"
  input: "^Adam/update_convolutional1/weight_matrix_conv1/ApplyAdam"
  input: "^Adam/update_convolutional1/bias_vector_conv1/ApplyAdam"
  input: "^Adam/update_convolutional2/weight_matrix_conv2/ApplyAdam"
  input: "^Adam/update_convolutional2/bias_vector_conv2/ApplyAdam"
  input: "^Adam/update_fully_connected_hidden1/weight_matrix_fc1/ApplyAdam"
  input: "^Adam/update_fully_connected_hidden1/bias_vector_fc1/ApplyAdam"
  input: "^Adam/update_fully_connected_output/weight_matrix_fc2/ApplyAdam"
  input: "^Adam/update_fully_connected_output/bias_vector_fc2/ApplyAdam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
}
node {
  name: "Adam/Assign_1"
  op: "Assign"
  input: "beta2_power"
  input: "Adam/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Adam"
  op: "NoOp"
  input: "^Adam/update_convolutional1/weight_matrix_conv1/ApplyAdam"
  input: "^Adam/update_convolutional1/bias_vector_conv1/ApplyAdam"
  input: "^Adam/update_convolutional2/weight_matrix_conv2/ApplyAdam"
  input: "^Adam/update_convolutional2/bias_vector_conv2/ApplyAdam"
  input: "^Adam/update_fully_connected_hidden1/weight_matrix_fc1/ApplyAdam"
  input: "^Adam/update_fully_connected_hidden1/bias_vector_fc1/ApplyAdam"
  input: "^Adam/update_fully_connected_output/weight_matrix_fc2/ApplyAdam"
  input: "^Adam/update_fully_connected_output/bias_vector_fc2/ApplyAdam"
  input: "^Adam/Assign"
  input: "^Adam/Assign_1"
}
node {
  name: "ArgMax/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ArgMax"
  op: "ArgMax"
  input: "fully_connected_output/output_node"
  input: "ArgMax/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ArgMax_1/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ArgMax_1"
  op: "ArgMax"
  input: "targets_matrix"
  input: "ArgMax_1/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Equal"
  op: "Equal"
  input: "ArgMax"
  input: "ArgMax_1"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "Cast"
  op: "Cast"
  input: "Equal"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Mean_1"
  op: "Mean"
  input: "Cast"
  input: "Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "init"
  op: "NoOp"
  input: "^convolutional1/weight_matrix_conv1/Assign"
  input: "^convolutional1/bias_vector_conv1/Assign"
  input: "^convolutional2/weight_matrix_conv2/Assign"
  input: "^convolutional2/bias_vector_conv2/Assign"
  input: "^fully_connected_hidden1/weight_matrix_fc1/Assign"
  input: "^fully_connected_hidden1/bias_vector_fc1/Assign"
  input: "^fully_connected_output/weight_matrix_fc2/Assign"
  input: "^fully_connected_output/bias_vector_fc2/Assign"
  input: "^beta1_power/Assign"
  input: "^beta2_power/Assign"
  input: "^convolutional1/weight_matrix_conv1/Adam/Assign"
  input: "^convolutional1/weight_matrix_conv1/Adam_1/Assign"
  input: "^convolutional1/bias_vector_conv1/Adam/Assign"
  input: "^convolutional1/bias_vector_conv1/Adam_1/Assign"
  input: "^convolutional2/weight_matrix_conv2/Adam/Assign"
  input: "^convolutional2/weight_matrix_conv2/Adam_1/Assign"
  input: "^convolutional2/bias_vector_conv2/Adam/Assign"
  input: "^convolutional2/bias_vector_conv2/Adam_1/Assign"
  input: "^fully_connected_hidden1/weight_matrix_fc1/Adam/Assign"
  input: "^fully_connected_hidden1/weight_matrix_fc1/Adam_1/Assign"
  input: "^fully_connected_hidden1/bias_vector_fc1/Adam/Assign"
  input: "^fully_connected_hidden1/bias_vector_fc1/Adam_1/Assign"
  input: "^fully_connected_output/weight_matrix_fc2/Adam/Assign"
  input: "^fully_connected_output/weight_matrix_fc2/Adam_1/Assign"
  input: "^fully_connected_output/bias_vector_fc2/Adam/Assign"
  input: "^fully_connected_output/bias_vector_fc2/Adam_1/Assign"
}
node {
  name: "save/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "model"
      }
    }
  }
}
node {
  name: "save/save/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 26
          }
        }
        string_val: "beta1_power"
        string_val: "beta2_power"
        string_val: "convolutional1/bias_vector_conv1"
        string_val: "convolutional1/bias_vector_conv1/Adam"
        string_val: "convolutional1/bias_vector_conv1/Adam_1"
        string_val: "convolutional1/weight_matrix_conv1"
        string_val: "convolutional1/weight_matrix_conv1/Adam"
        string_val: "convolutional1/weight_matrix_conv1/Adam_1"
        string_val: "convolutional2/bias_vector_conv2"
        string_val: "convolutional2/bias_vector_conv2/Adam"
        string_val: "convolutional2/bias_vector_conv2/Adam_1"
        string_val: "convolutional2/weight_matrix_conv2"
        string_val: "convolutional2/weight_matrix_conv2/Adam"
        string_val: "convolutional2/weight_matrix_conv2/Adam_1"
        string_val: "fully_connected_hidden1/bias_vector_fc1"
        string_val: "fully_connected_hidden1/bias_vector_fc1/Adam"
        string_val: "fully_connected_hidden1/bias_vector_fc1/Adam_1"
        string_val: "fully_connected_hidden1/weight_matrix_fc1"
        string_val: "fully_connected_hidden1/weight_matrix_fc1/Adam"
        string_val: "fully_connected_hidden1/weight_matrix_fc1/Adam_1"
        string_val: "fully_connected_output/bias_vector_fc2"
        string_val: "fully_connected_output/bias_vector_fc2/Adam"
        string_val: "fully_connected_output/bias_vector_fc2/Adam_1"
        string_val: "fully_connected_output/weight_matrix_fc2"
        string_val: "fully_connected_output/weight_matrix_fc2/Adam"
        string_val: "fully_connected_output/weight_matrix_fc2/Adam_1"
      }
    }
  }
}
node {
  name: "save/save/shapes_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 26
          }
        }
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
      }
    }
  }
}
node {
  name: "save/save"
  op: "SaveSlices"
  input: "save/Const"
  input: "save/save/tensor_names"
  input: "save/save/shapes_and_slices"
  input: "beta1_power"
  input: "beta2_power"
  input: "convolutional1/bias_vector_conv1"
  input: "convolutional1/bias_vector_conv1/Adam"
  input: "convolutional1/bias_vector_conv1/Adam_1"
  input: "convolutional1/weight_matrix_conv1"
  input: "convolutional1/weight_matrix_conv1/Adam"
  input: "convolutional1/weight_matrix_conv1/Adam_1"
  input: "convolutional2/bias_vector_conv2"
  input: "convolutional2/bias_vector_conv2/Adam"
  input: "convolutional2/bias_vector_conv2/Adam_1"
  input: "convolutional2/weight_matrix_conv2"
  input: "convolutional2/weight_matrix_conv2/Adam"
  input: "convolutional2/weight_matrix_conv2/Adam_1"
  input: "fully_connected_hidden1/bias_vector_fc1"
  input: "fully_connected_hidden1/bias_vector_fc1/Adam"
  input: "fully_connected_hidden1/bias_vector_fc1/Adam_1"
  input: "fully_connected_hidden1/weight_matrix_fc1"
  input: "fully_connected_hidden1/weight_matrix_fc1/Adam"
  input: "fully_connected_hidden1/weight_matrix_fc1/Adam_1"
  input: "fully_connected_output/bias_vector_fc2"
  input: "fully_connected_output/bias_vector_fc2/Adam"
  input: "fully_connected_output/bias_vector_fc2/Adam_1"
  input: "fully_connected_output/weight_matrix_fc2"
  input: "fully_connected_output/weight_matrix_fc2/Adam"
  input: "fully_connected_output/weight_matrix_fc2/Adam_1"
  attr {
    key: "T"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/control_dependency"
  op: "Identity"
  input: "save/Const"
  input: "^save/save"
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@save/Const"
      }
    }
  }
}
node {
  name: "save/restore_slice/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "beta1_power"
      }
    }
  }
}
node {
  name: "save/restore_slice/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice/tensor_name"
  input: "save/restore_slice/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign"
  op: "Assign"
  input: "beta1_power"
  input: "save/restore_slice"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_1/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "beta2_power"
      }
    }
  }
}
node {
  name: "save/restore_slice_1/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_1"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_1/tensor_name"
  input: "save/restore_slice_1/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_1"
  op: "Assign"
  input: "beta2_power"
  input: "save/restore_slice_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_2/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional1/bias_vector_conv1"
      }
    }
  }
}
node {
  name: "save/restore_slice_2/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_2"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_2/tensor_name"
  input: "save/restore_slice_2/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_2"
  op: "Assign"
  input: "convolutional1/bias_vector_conv1"
  input: "save/restore_slice_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_3/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional1/bias_vector_conv1/Adam"
      }
    }
  }
}
node {
  name: "save/restore_slice_3/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_3"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_3/tensor_name"
  input: "save/restore_slice_3/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_3"
  op: "Assign"
  input: "convolutional1/bias_vector_conv1/Adam"
  input: "save/restore_slice_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_4/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional1/bias_vector_conv1/Adam_1"
      }
    }
  }
}
node {
  name: "save/restore_slice_4/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_4"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_4/tensor_name"
  input: "save/restore_slice_4/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_4"
  op: "Assign"
  input: "convolutional1/bias_vector_conv1/Adam_1"
  input: "save/restore_slice_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/bias_vector_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_5/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional1/weight_matrix_conv1"
      }
    }
  }
}
node {
  name: "save/restore_slice_5/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_5"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_5/tensor_name"
  input: "save/restore_slice_5/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_5"
  op: "Assign"
  input: "convolutional1/weight_matrix_conv1"
  input: "save/restore_slice_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_6/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional1/weight_matrix_conv1/Adam"
      }
    }
  }
}
node {
  name: "save/restore_slice_6/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_6"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_6/tensor_name"
  input: "save/restore_slice_6/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_6"
  op: "Assign"
  input: "convolutional1/weight_matrix_conv1/Adam"
  input: "save/restore_slice_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_7/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional1/weight_matrix_conv1/Adam_1"
      }
    }
  }
}
node {
  name: "save/restore_slice_7/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_7"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_7/tensor_name"
  input: "save/restore_slice_7/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_7"
  op: "Assign"
  input: "convolutional1/weight_matrix_conv1/Adam_1"
  input: "save/restore_slice_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional1/weight_matrix_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_8/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional2/bias_vector_conv2"
      }
    }
  }
}
node {
  name: "save/restore_slice_8/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_8"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_8/tensor_name"
  input: "save/restore_slice_8/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_8"
  op: "Assign"
  input: "convolutional2/bias_vector_conv2"
  input: "save/restore_slice_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_9/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional2/bias_vector_conv2/Adam"
      }
    }
  }
}
node {
  name: "save/restore_slice_9/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_9"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_9/tensor_name"
  input: "save/restore_slice_9/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_9"
  op: "Assign"
  input: "convolutional2/bias_vector_conv2/Adam"
  input: "save/restore_slice_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_10/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional2/bias_vector_conv2/Adam_1"
      }
    }
  }
}
node {
  name: "save/restore_slice_10/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_10"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_10/tensor_name"
  input: "save/restore_slice_10/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_10"
  op: "Assign"
  input: "convolutional2/bias_vector_conv2/Adam_1"
  input: "save/restore_slice_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/bias_vector_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_11/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional2/weight_matrix_conv2"
      }
    }
  }
}
node {
  name: "save/restore_slice_11/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_11"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_11/tensor_name"
  input: "save/restore_slice_11/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_11"
  op: "Assign"
  input: "convolutional2/weight_matrix_conv2"
  input: "save/restore_slice_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_12/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional2/weight_matrix_conv2/Adam"
      }
    }
  }
}
node {
  name: "save/restore_slice_12/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_12"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_12/tensor_name"
  input: "save/restore_slice_12/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_12"
  op: "Assign"
  input: "convolutional2/weight_matrix_conv2/Adam"
  input: "save/restore_slice_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_13/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "convolutional2/weight_matrix_conv2/Adam_1"
      }
    }
  }
}
node {
  name: "save/restore_slice_13/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_13"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_13/tensor_name"
  input: "save/restore_slice_13/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_13"
  op: "Assign"
  input: "convolutional2/weight_matrix_conv2/Adam_1"
  input: "save/restore_slice_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@convolutional2/weight_matrix_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_14/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
}
node {
  name: "save/restore_slice_14/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_14"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_14/tensor_name"
  input: "save/restore_slice_14/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_14"
  op: "Assign"
  input: "fully_connected_hidden1/bias_vector_fc1"
  input: "save/restore_slice_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_15/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_hidden1/bias_vector_fc1/Adam"
      }
    }
  }
}
node {
  name: "save/restore_slice_15/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_15"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_15/tensor_name"
  input: "save/restore_slice_15/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_15"
  op: "Assign"
  input: "fully_connected_hidden1/bias_vector_fc1/Adam"
  input: "save/restore_slice_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_16/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_hidden1/bias_vector_fc1/Adam_1"
      }
    }
  }
}
node {
  name: "save/restore_slice_16/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_16"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_16/tensor_name"
  input: "save/restore_slice_16/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_16"
  op: "Assign"
  input: "fully_connected_hidden1/bias_vector_fc1/Adam_1"
  input: "save/restore_slice_16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/bias_vector_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_17/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
}
node {
  name: "save/restore_slice_17/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_17"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_17/tensor_name"
  input: "save/restore_slice_17/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_17"
  op: "Assign"
  input: "fully_connected_hidden1/weight_matrix_fc1"
  input: "save/restore_slice_17"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_18/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_hidden1/weight_matrix_fc1/Adam"
      }
    }
  }
}
node {
  name: "save/restore_slice_18/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_18"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_18/tensor_name"
  input: "save/restore_slice_18/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_18"
  op: "Assign"
  input: "fully_connected_hidden1/weight_matrix_fc1/Adam"
  input: "save/restore_slice_18"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_19/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_hidden1/weight_matrix_fc1/Adam_1"
      }
    }
  }
}
node {
  name: "save/restore_slice_19/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_19"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_19/tensor_name"
  input: "save/restore_slice_19/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_19"
  op: "Assign"
  input: "fully_connected_hidden1/weight_matrix_fc1/Adam_1"
  input: "save/restore_slice_19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_hidden1/weight_matrix_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_20/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_output/bias_vector_fc2"
      }
    }
  }
}
node {
  name: "save/restore_slice_20/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_20"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_20/tensor_name"
  input: "save/restore_slice_20/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_20"
  op: "Assign"
  input: "fully_connected_output/bias_vector_fc2"
  input: "save/restore_slice_20"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_21/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_output/bias_vector_fc2/Adam"
      }
    }
  }
}
node {
  name: "save/restore_slice_21/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_21"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_21/tensor_name"
  input: "save/restore_slice_21/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_21"
  op: "Assign"
  input: "fully_connected_output/bias_vector_fc2/Adam"
  input: "save/restore_slice_21"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_22/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_output/bias_vector_fc2/Adam_1"
      }
    }
  }
}
node {
  name: "save/restore_slice_22/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_22"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_22/tensor_name"
  input: "save/restore_slice_22/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_22"
  op: "Assign"
  input: "fully_connected_output/bias_vector_fc2/Adam_1"
  input: "save/restore_slice_22"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/bias_vector_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_23/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_output/weight_matrix_fc2"
      }
    }
  }
}
node {
  name: "save/restore_slice_23/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_23"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_23/tensor_name"
  input: "save/restore_slice_23/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_23"
  op: "Assign"
  input: "fully_connected_output/weight_matrix_fc2"
  input: "save/restore_slice_23"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_24/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_output/weight_matrix_fc2/Adam"
      }
    }
  }
}
node {
  name: "save/restore_slice_24/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_24"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_24/tensor_name"
  input: "save/restore_slice_24/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_24"
  op: "Assign"
  input: "fully_connected_output/weight_matrix_fc2/Adam"
  input: "save/restore_slice_24"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_slice_25/tensor_name"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "fully_connected_output/weight_matrix_fc2/Adam_1"
      }
    }
  }
}
node {
  name: "save/restore_slice_25/shape_and_slice"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/restore_slice_25"
  op: "RestoreSlice"
  input: "save/Const"
  input: "save/restore_slice_25/tensor_name"
  input: "save/restore_slice_25/shape_and_slice"
  attr {
    key: "dt"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "preferred_shard"
    value {
      i: -1
    }
  }
}
node {
  name: "save/Assign_25"
  op: "Assign"
  input: "fully_connected_output/weight_matrix_fc2/Adam_1"
  input: "save/restore_slice_25"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@fully_connected_output/weight_matrix_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_all"
  op: "NoOp"
  input: "^save/Assign"
  input: "^save/Assign_1"
  input: "^save/Assign_2"
  input: "^save/Assign_3"
  input: "^save/Assign_4"
  input: "^save/Assign_5"
  input: "^save/Assign_6"
  input: "^save/Assign_7"
  input: "^save/Assign_8"
  input: "^save/Assign_9"
  input: "^save/Assign_10"
  input: "^save/Assign_11"
  input: "^save/Assign_12"
  input: "^save/Assign_13"
  input: "^save/Assign_14"
  input: "^save/Assign_15"
  input: "^save/Assign_16"
  input: "^save/Assign_17"
  input: "^save/Assign_18"
  input: "^save/Assign_19"
  input: "^save/Assign_20"
  input: "^save/Assign_21"
  input: "^save/Assign_22"
  input: "^save/Assign_23"
  input: "^save/Assign_24"
  input: "^save/Assign_25"
}
versions {
  producer: 10
}
