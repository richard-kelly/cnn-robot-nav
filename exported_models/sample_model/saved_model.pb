¤
đ
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
ź
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ě
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
Ô
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.9.02b'v1.9.0-0-g25c197e023'ĺŘ

global_step/Initializer/zerosConst*
value	B	 R *
_output_shapes
: *
dtype0	*
_class
loc:@global_step

global_step
VariableV2*
shared_name *
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
	container *
shape: 
˛
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_output_shapes
: *
_class
loc:@global_step
f
PlaceholderPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_1Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
f
Reshape/shapeConst*%
valueB"˙˙˙˙   8     *
_output_shapes
:*
dtype0
w
ReshapeReshapePlaceholderReshape/shape*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
`
Reshape_1/shapeConst*
valueB"˙˙˙˙   *
_output_shapes
:*
dtype0
t
	Reshape_1ReshapePlaceholder_1Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"         @   *
_output_shapes
:*
dtype0* 
_class
loc:@conv2d/kernel

,conv2d/kernel/Initializer/random_uniform/minConst*
valueB 2˝OŔe˝ż*
_output_shapes
: *
dtype0* 
_class
loc:@conv2d/kernel

,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB 2˝OŔe˝?*
_output_shapes
: *
dtype0* 
_class
loc:@conv2d/kernel
đ
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
seed2 * 
_class
loc:@conv2d/kernel*
T0*&
_output_shapes
:@*
dtype0*

seed 
Ň
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: * 
_class
loc:@conv2d/kernel
ě
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0*&
_output_shapes
:@* 
_class
loc:@conv2d/kernel
Ţ
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0*&
_output_shapes
:@* 
_class
loc:@conv2d/kernel
ł
conv2d/kernel
VariableV2*
shared_name * 
_class
loc:@conv2d/kernel*&
_output_shapes
:@*
dtype0*
	container *
shape:@
Ó
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
T0*&
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel

conv2d/kernel/readIdentityconv2d/kernel*
T0*&
_output_shapes
:@* 
_class
loc:@conv2d/kernel

conv2d/bias/Initializer/zerosConst*
valueB@2        *
_output_shapes
:@*
dtype0*
_class
loc:@conv2d/bias

conv2d/bias
VariableV2*
shared_name *
_class
loc:@conv2d/bias*
_output_shapes
:@*
dtype0*
	container *
shape:@
ś
conv2d/bias/AssignAssignconv2d/biasconv2d/bias/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
n
conv2d/bias/readIdentityconv2d/bias*
T0*
_output_shapes
:@*
_class
loc:@conv2d/bias
e
conv2d/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ý
conv2d/Conv2DConv2DReshapeconv2d/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙č@*
use_cudnn_on_gpu(

conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/bias/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙č@
^
conv2d/ReluReluconv2d/BiasAdd*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙č@
ş
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
strides
*
ksize
*
T0*
data_formatNHWC*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0*"
_class
loc:@conv2d_1/kernel

.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB 2      Ŕż*
_output_shapes
: *
dtype0*"
_class
loc:@conv2d_1/kernel

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB 2      Ŕ?*
_output_shapes
: *
dtype0*"
_class
loc:@conv2d_1/kernel
ö
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
seed2 *"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
:@@*
dtype0*

seed 
Ú
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
ô
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_1/kernel
ć
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_1/kernel
ˇ
conv2d_1/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@*
dtype0*
	container *
shape:@@
Ű
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel

conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_1/kernel

conv2d_1/bias/Initializer/zerosConst*
valueB@2        *
_output_shapes
:@*
dtype0* 
_class
loc:@conv2d_1/bias

conv2d_1/bias
VariableV2*
shared_name * 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
dtype0*
	container *
shape:@
ž
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/bias/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
g
conv2d_1/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
î
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@*
use_cudnn_on_gpu(

conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
a
conv2d_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
­
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0*"
_class
loc:@conv2d_2/kernel

.conv2d_2/kernel/Initializer/random_uniform/minConst*
valueB 2      Ŕż*
_output_shapes
: *
dtype0*"
_class
loc:@conv2d_2/kernel

.conv2d_2/kernel/Initializer/random_uniform/maxConst*
valueB 2      Ŕ?*
_output_shapes
: *
dtype0*"
_class
loc:@conv2d_2/kernel
ö
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
seed2 *"
_class
loc:@conv2d_2/kernel*
T0*&
_output_shapes
:@@*
dtype0*

seed 
Ú
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel
ô
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
ć
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
ˇ
conv2d_2/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
dtype0*
	container *
shape:@@
Ű
conv2d_2/kernel/AssignAssignconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_2/kernel

conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel

conv2d_2/bias/Initializer/zerosConst*
valueB@2        *
_output_shapes
:@*
dtype0* 
_class
loc:@conv2d_2/bias

conv2d_2/bias
VariableV2*
shared_name * 
_class
loc:@conv2d_2/bias*
_output_shapes
:@*
dtype0*
	container *
shape:@
ž
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/bias/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_2/bias
t
conv2d_2/bias/readIdentityconv2d_2/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
g
conv2d_2/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
ć
conv2d_2/Conv2DConv2Dconv2d_1/Reluconv2d_2/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@*
use_cudnn_on_gpu(

conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
m
addAddmax_pooling2d/MaxPoolconv2d_2/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
K
ReluReluadd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
­
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0*"
_class
loc:@conv2d_3/kernel

.conv2d_3/kernel/Initializer/random_uniform/minConst*
valueB 2      Ŕż*
_output_shapes
: *
dtype0*"
_class
loc:@conv2d_3/kernel

.conv2d_3/kernel/Initializer/random_uniform/maxConst*
valueB 2      Ŕ?*
_output_shapes
: *
dtype0*"
_class
loc:@conv2d_3/kernel
ö
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
seed2 *"
_class
loc:@conv2d_3/kernel*
T0*&
_output_shapes
:@@*
dtype0*

seed 
Ú
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel
ô
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_3/kernel
ć
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_3/kernel
ˇ
conv2d_3/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:@@*
dtype0*
	container *
shape:@@
Ű
conv2d_3/kernel/AssignAssignconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_3/kernel

conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_3/kernel

conv2d_3/bias/Initializer/zerosConst*
valueB@2        *
_output_shapes
:@*
dtype0* 
_class
loc:@conv2d_3/bias

conv2d_3/bias
VariableV2*
shared_name * 
_class
loc:@conv2d_3/bias*
_output_shapes
:@*
dtype0*
	container *
shape:@
ž
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/bias/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_3/bias
t
conv2d_3/bias/readIdentityconv2d_3/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_3/bias
g
conv2d_3/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
Ý
conv2d_3/Conv2DConv2DReluconv2d_3/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@*
use_cudnn_on_gpu(

conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_3/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
a
conv2d_3/ReluReluconv2d_3/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
­
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0*"
_class
loc:@conv2d_4/kernel

.conv2d_4/kernel/Initializer/random_uniform/minConst*
valueB 2      Ŕż*
_output_shapes
: *
dtype0*"
_class
loc:@conv2d_4/kernel

.conv2d_4/kernel/Initializer/random_uniform/maxConst*
valueB 2      Ŕ?*
_output_shapes
: *
dtype0*"
_class
loc:@conv2d_4/kernel
ö
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
seed2 *"
_class
loc:@conv2d_4/kernel*
T0*&
_output_shapes
:@@*
dtype0*

seed 
Ú
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel
ô
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_4/kernel
ć
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_4/kernel
ˇ
conv2d_4/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@@*
dtype0*
	container *
shape:@@
Ű
conv2d_4/kernel/AssignAssignconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_4/kernel

conv2d_4/kernel/readIdentityconv2d_4/kernel*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_4/kernel

conv2d_4/bias/Initializer/zerosConst*
valueB@2        *
_output_shapes
:@*
dtype0* 
_class
loc:@conv2d_4/bias

conv2d_4/bias
VariableV2*
shared_name * 
_class
loc:@conv2d_4/bias*
_output_shapes
:@*
dtype0*
	container *
shape:@
ž
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/bias/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_4/bias
t
conv2d_4/bias/readIdentityconv2d_4/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_4/bias
g
conv2d_4/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
ć
conv2d_4/Conv2DConv2Dconv2d_3/Reluconv2d_4/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@*
use_cudnn_on_gpu(

conv2d_4/BiasAddBiasAddconv2d_4/Conv2Dconv2d_4/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
j
add_1Addconv2d_2/BiasAddconv2d_4/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
O
Relu_1Reluadd_1*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x@
š
average_pooling2d/AvgPoolAvgPoolRelu_1*
strides
*
ksize
*
T0*
data_formatNHWC*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙(@
`
Reshape_2/shapeConst*
valueB"˙˙˙˙ 
  *
_output_shapes
:*
dtype0

	Reshape_2Reshapeaverage_pooling2d/AvgPoolReshape_2/shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
M
concat/axisConst*
value	B :*
_output_shapes
: *
dtype0
}
concatConcatV2	Reshape_2	Reshape_1concat/axis*
T0*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"
     *
_output_shapes
:*
dtype0*
_class
loc:@dense/kernel

+dense/kernel/Initializer/random_uniform/minConst*
valueB 2ź4Žđ¤ż*
_output_shapes
: *
dtype0*
_class
loc:@dense/kernel

+dense/kernel/Initializer/random_uniform/maxConst*
valueB 2ź4Žđ¤?*
_output_shapes
: *
dtype0*
_class
loc:@dense/kernel
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
seed2 *
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
*
dtype0*

seed 
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *
_class
loc:@dense/kernel
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
*
_class
loc:@dense/kernel
Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*
_class
loc:@dense/kernel
Ľ
dense/kernel
VariableV2*
shared_name *
_class
loc:@dense/kernel* 
_output_shapes
:
*
dtype0*
	container *
shape:

É
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
w
dense/kernel/readIdentitydense/kernel*
T0* 
_output_shapes
:
*
_class
loc:@dense/kernel

,dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*
_output_shapes
:*
dtype0*
_class
loc:@dense/bias

"dense/bias/Initializer/zeros/ConstConst*
valueB 2        *
_output_shapes
: *
dtype0*
_class
loc:@dense/bias
Í
dense/bias/Initializer/zerosFill,dense/bias/Initializer/zeros/shape_as_tensor"dense/bias/Initializer/zeros/Const*
T0*
_output_shapes	
:*

index_type0*
_class
loc:@dense/bias


dense/bias
VariableV2*
shared_name *
_class
loc:@dense/bias*
_output_shapes	
:*
dtype0*
	container *
shape:
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
l
dense/bias/readIdentity
dense/bias*
T0*
_output_shapes	
:*
_class
loc:@dense/bias

dense/MatMulMatMulconcatdense/kernel/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T

dense/ReluReludense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*
dtype0*!
_class
loc:@dense_1/kernel

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB 2ŞLXčzśŤż*
_output_shapes
: *
dtype0*!
_class
loc:@dense_1/kernel

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB 2ŞLXčzśŤ?*
_output_shapes
: *
dtype0*!
_class
loc:@dense_1/kernel
í
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
seed2 *!
_class
loc:@dense_1/kernel*
T0* 
_output_shapes
:
*
dtype0*

seed 
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *!
_class
loc:@dense_1/kernel
ę
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel
Ü
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel
Š
dense_1/kernel
VariableV2*
shared_name *!
_class
loc:@dense_1/kernel* 
_output_shapes
:
*
dtype0*
	container *
shape:

Ń
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel
}
dense_1/kernel/readIdentitydense_1/kernel*
T0* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel

.dense_1/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*
_output_shapes
:*
dtype0*
_class
loc:@dense_1/bias

$dense_1/bias/Initializer/zeros/ConstConst*
valueB 2        *
_output_shapes
: *
dtype0*
_class
loc:@dense_1/bias
Ő
dense_1/bias/Initializer/zerosFill.dense_1/bias/Initializer/zeros/shape_as_tensor$dense_1/bias/Initializer/zeros/Const*
T0*
_output_shapes	
:*

index_type0*
_class
loc:@dense_1/bias

dense_1/bias
VariableV2*
shared_name *
_class
loc:@dense_1/bias*
_output_shapes	
:*
dtype0*
	container *
shape:
ť
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_output_shapes	
:*
_class
loc:@dense_1/bias

dense_1/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
dense_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*
dtype0*!
_class
loc:@dense_2/kernel

-dense_2/kernel/Initializer/random_uniform/minConst*
valueB 2      °ż*
_output_shapes
: *
dtype0*!
_class
loc:@dense_2/kernel

-dense_2/kernel/Initializer/random_uniform/maxConst*
valueB 2      °?*
_output_shapes
: *
dtype0*!
_class
loc:@dense_2/kernel
í
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*
seed2 *!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:
*
dtype0*

seed 
Ö
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *!
_class
loc:@dense_2/kernel
ę
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
*!
_class
loc:@dense_2/kernel
Ü
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*!
_class
loc:@dense_2/kernel
Š
dense_2/kernel
VariableV2*
shared_name *!
_class
loc:@dense_2/kernel* 
_output_shapes
:
*
dtype0*
	container *
shape:

Ń
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*!
_class
loc:@dense_2/kernel
}
dense_2/kernel/readIdentitydense_2/kernel*
T0* 
_output_shapes
:
*!
_class
loc:@dense_2/kernel

dense_2/bias/Initializer/zerosConst*
valueB2        *
_output_shapes	
:*
dtype0*
_class
loc:@dense_2/bias

dense_2/bias
VariableV2*
shared_name *
_class
loc:@dense_2/bias*
_output_shapes	
:*
dtype0*
	container *
shape:
ť
dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/zeros*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*
_class
loc:@dense_2/bias
r
dense_2/bias/readIdentitydense_2/bias*
T0*
_output_shapes	
:*
_class
loc:@dense_2/bias

dense_2/MatMulMatMuldense_1/Reludense_2/kernel/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
dense_2/ReluReludense_2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_3/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*
dtype0*!
_class
loc:@dense_3/kernel

-dense_3/kernel/Initializer/random_uniform/minConst*
valueB 2Wž°Ş¨ťż*
_output_shapes
: *
dtype0*!
_class
loc:@dense_3/kernel

-dense_3/kernel/Initializer/random_uniform/maxConst*
valueB 2Wž°Ş¨ť?*
_output_shapes
: *
dtype0*!
_class
loc:@dense_3/kernel
ě
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
seed2 *!
_class
loc:@dense_3/kernel*
T0*
_output_shapes
:	*
dtype0*

seed 
Ö
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *!
_class
loc:@dense_3/kernel
é
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	*!
_class
loc:@dense_3/kernel
Ű
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	*!
_class
loc:@dense_3/kernel
§
dense_3/kernel
VariableV2*
shared_name *!
_class
loc:@dense_3/kernel*
_output_shapes
:	*
dtype0*
	container *
shape:	
Đ
dense_3/kernel/AssignAssigndense_3/kernel)dense_3/kernel/Initializer/random_uniform*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*!
_class
loc:@dense_3/kernel
|
dense_3/kernel/readIdentitydense_3/kernel*
T0*
_output_shapes
:	*!
_class
loc:@dense_3/kernel

dense_3/bias/Initializer/zerosConst*
valueB2        *
_output_shapes
:*
dtype0*
_class
loc:@dense_3/bias

dense_3/bias
VariableV2*
shared_name *
_class
loc:@dense_3/bias*
_output_shapes
:*
dtype0*
	container *
shape:
ş
dense_3/bias/AssignAssigndense_3/biasdense_3/bias/Initializer/zeros*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@dense_3/bias
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_output_shapes
:*
_class
loc:@dense_3/bias

dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
P

save/ConstConst*
valueB Bmodel*
_output_shapes
: *
dtype0

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_8386b52fb8924cc2afe74b8ac2214d49/part*
_output_shapes
: *
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 

save/SaveV2/tensor_namesConst"/device:CPU:0*˛
value¨BĽBconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBconv2d_3/biasBconv2d_3/kernelBconv2d_4/biasBconv2d_4/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBglobal_step*
_output_shapes
:*
dtype0

save/SaveV2/shape_and_slicesConst"/device:CPU:0*9
value0B.B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0
˛
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernelconv2d_2/biasconv2d_2/kernelconv2d_3/biasconv2d_3/kernelconv2d_4/biasconv2d_4/kernel
dense/biasdense/kerneldense_1/biasdense_1/kerneldense_2/biasdense_2/kerneldense_3/biasdense_3/kernelglobal_step"/device:CPU:0*!
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename
Ź
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*

axis *
T0*
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 

save/RestoreV2/tensor_namesConst"/device:CPU:0*˛
value¨BĽBconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBconv2d_3/biasBconv2d_3/kernelBconv2d_4/biasBconv2d_4/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBglobal_step*
_output_shapes
:*
dtype0

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*9
value0B.B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0
ů
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*!
dtypes
2	*`
_output_shapesN
L:::::::::::::::::::
 
save/AssignAssignconv2d/biassave/RestoreV2*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
´
save/Assign_1Assignconv2d/kernelsave/RestoreV2:1*
T0*&
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
¨
save/Assign_2Assignconv2d_1/biassave/RestoreV2:2*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
¸
save/Assign_3Assignconv2d_1/kernelsave/RestoreV2:3*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
¨
save/Assign_4Assignconv2d_2/biassave/RestoreV2:4*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_2/bias
¸
save/Assign_5Assignconv2d_2/kernelsave/RestoreV2:5*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_2/kernel
¨
save/Assign_6Assignconv2d_3/biassave/RestoreV2:6*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_3/bias
¸
save/Assign_7Assignconv2d_3/kernelsave/RestoreV2:7*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_3/kernel
¨
save/Assign_8Assignconv2d_4/biassave/RestoreV2:8*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_4/bias
¸
save/Assign_9Assignconv2d_4/kernelsave/RestoreV2:9*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_4/kernel
Ľ
save/Assign_10Assign
dense/biassave/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
Ž
save/Assign_11Assigndense/kernelsave/RestoreV2:11*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
Š
save/Assign_12Assigndense_1/biassave/RestoreV2:12*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias
˛
save/Assign_13Assigndense_1/kernelsave/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel
Š
save/Assign_14Assigndense_2/biassave/RestoreV2:14*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*
_class
loc:@dense_2/bias
˛
save/Assign_15Assigndense_2/kernelsave/RestoreV2:15*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*!
_class
loc:@dense_2/kernel
¨
save/Assign_16Assigndense_3/biassave/RestoreV2:16*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@dense_3/bias
ą
save/Assign_17Assigndense_3/kernelsave/RestoreV2:17*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*!
_class
loc:@dense_3/kernel
˘
save/Assign_18Assignglobal_stepsave/RestoreV2:18*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@global_step
Ń
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
R
save_1/ConstConst*
valueB Bmodel*
_output_shapes
: *
dtype0

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_b882fedd691a41d4969a35bbe066e732/part*
_output_shapes
: *
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 

save_1/SaveV2/tensor_namesConst"/device:CPU:0*˛
value¨BĽBconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBconv2d_3/biasBconv2d_3/kernelBconv2d_4/biasBconv2d_4/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBglobal_step*
_output_shapes
:*
dtype0

save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*9
value0B.B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0
ş
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernelconv2d_2/biasconv2d_2/kernelconv2d_3/biasconv2d_3/kernelconv2d_4/biasconv2d_4/kernel
dense/biasdense/kerneldense_1/biasdense_1/kerneldense_2/biasdense_2/kerneldense_3/biasdense_3/kernelglobal_step"/device:CPU:0*!
dtypes
2	
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename
˛
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*

axis *
T0*
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 

save_1/RestoreV2/tensor_namesConst"/device:CPU:0*˛
value¨BĽBconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBconv2d_3/biasBconv2d_3/kernelBconv2d_4/biasBconv2d_4/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBglobal_step*
_output_shapes
:*
dtype0

!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*9
value0B.B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*!
dtypes
2	*`
_output_shapesN
L:::::::::::::::::::
¤
save_1/AssignAssignconv2d/biassave_1/RestoreV2*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
¸
save_1/Assign_1Assignconv2d/kernelsave_1/RestoreV2:1*
T0*&
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
Ź
save_1/Assign_2Assignconv2d_1/biassave_1/RestoreV2:2*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
ź
save_1/Assign_3Assignconv2d_1/kernelsave_1/RestoreV2:3*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
Ź
save_1/Assign_4Assignconv2d_2/biassave_1/RestoreV2:4*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_2/bias
ź
save_1/Assign_5Assignconv2d_2/kernelsave_1/RestoreV2:5*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_2/kernel
Ź
save_1/Assign_6Assignconv2d_3/biassave_1/RestoreV2:6*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_3/bias
ź
save_1/Assign_7Assignconv2d_3/kernelsave_1/RestoreV2:7*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_3/kernel
Ź
save_1/Assign_8Assignconv2d_4/biassave_1/RestoreV2:8*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_4/bias
ź
save_1/Assign_9Assignconv2d_4/kernelsave_1/RestoreV2:9*
T0*&
_output_shapes
:@@*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_4/kernel
Š
save_1/Assign_10Assign
dense/biassave_1/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
˛
save_1/Assign_11Assigndense/kernelsave_1/RestoreV2:11*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
­
save_1/Assign_12Assigndense_1/biassave_1/RestoreV2:12*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias
ś
save_1/Assign_13Assigndense_1/kernelsave_1/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel
­
save_1/Assign_14Assigndense_2/biassave_1/RestoreV2:14*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*
_class
loc:@dense_2/bias
ś
save_1/Assign_15Assigndense_2/kernelsave_1/RestoreV2:15*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*!
_class
loc:@dense_2/kernel
Ź
save_1/Assign_16Assigndense_3/biassave_1/RestoreV2:16*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@dense_3/bias
ľ
save_1/Assign_17Assigndense_3/kernelsave_1/RestoreV2:17*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*!
_class
loc:@dense_3/kernel
Ś
save_1/Assign_18Assignglobal_stepsave_1/RestoreV2:18*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@global_step
ů
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8" 
legacy_init_op


group_deps"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"ü
trainable_variablesäá
k
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:08
Z
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:02conv2d/bias/Initializer/zeros:08
s
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:08
b
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02!conv2d_1/bias/Initializer/zeros:08
s
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02,conv2d_2/kernel/Initializer/random_uniform:08
b
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02!conv2d_2/bias/Initializer/zeros:08
s
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02,conv2d_3/kernel/Initializer/random_uniform:08
b
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02!conv2d_3/bias/Initializer/zeros:08
s
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02,conv2d_4/kernel/Initializer/random_uniform:08
b
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02!conv2d_4/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08
o
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02+dense_2/kernel/Initializer/random_uniform:08
^
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02 dense_2/bias/Initializer/zeros:08
o
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02+dense_3/kernel/Initializer/random_uniform:08
^
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02 dense_3/bias/Initializer/zeros:08"Ě
	variablesžť
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
k
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:08
Z
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:02conv2d/bias/Initializer/zeros:08
s
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:08
b
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02!conv2d_1/bias/Initializer/zeros:08
s
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02,conv2d_2/kernel/Initializer/random_uniform:08
b
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02!conv2d_2/bias/Initializer/zeros:08
s
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02,conv2d_3/kernel/Initializer/random_uniform:08
b
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02!conv2d_3/bias/Initializer/zeros:08
s
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02,conv2d_4/kernel/Initializer/random_uniform:08
b
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02!conv2d_4/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08
o
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02+dense_2/kernel/Initializer/random_uniform:08
^
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02 dense_2/bias/Initializer/zeros:08
o
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02+dense_3/kernel/Initializer/random_uniform:08
^
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02 dense_3/bias/Initializer/zeros:08*ź
serving_default¨
*
goal"
Placeholder_1:0˙˙˙˙˙˙˙˙˙
)
laser 
Placeholder:0˙˙˙˙˙˙˙˙˙3
cmd_vel(
dense_3/BiasAdd:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict*ť
predict_output¨
*
goal"
Placeholder_1:0˙˙˙˙˙˙˙˙˙
)
laser 
Placeholder:0˙˙˙˙˙˙˙˙˙3
cmd_vel(
dense_3/BiasAdd:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict