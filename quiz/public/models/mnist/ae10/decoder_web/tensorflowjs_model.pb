
>
Z_inPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

:
Reshape_5/shapeConst*
value
B*
dtype0
M
"ae/dec3_bn_1/FusedBatchNorm/OffsetConst*
value
B@*
dtype0
M
"ae/dec3_bn_1/FusedBatchNorm/ScaledConst*
dtype0*
value
B@
:
Reshape_4/shapeConst*
value
B*
dtype0
F
ae/dec2_bn_1/batchnorm/subConst*
valueB	1*
dtype0
F
ae/dec2_bn_1/batchnorm/mulConst*
valueB	1*
dtype0
8
ae/dec2/biasConst*
valueB	1*
dtype0
?
ae/dec2/kernelConst*
valueB
1*
dtype0
F
ae/dec1_bn_1/batchnorm/subConst*
valueB	*
dtype0
F
ae/dec1_bn_1/batchnorm/mulConst*
dtype0*
valueB	
8
ae/dec1/biasConst*
dtype0*
valueB	
>
ae/dec1/kernelConst*
valueB	
*
dtype0
F
ae/dec3/kernelConst*
dtype0* 
valueB@
8
ae/dec3_1/stack/3Const*
valueB *
dtype0
6
ae/dec4_1/mul/yConst*
dtype0*
valueB 
J
ae/dec4_1/strided_slice/stack_1Const*
value
B*
dtype0
L
!ae/dec4_1/strided_slice_2/stack_1Const*
dtype0*
value
B
L
!ae/dec4_1/strided_slice_1/stack_1Const*
value
B*
dtype0
H
ae/dec4_1/strided_slice/stackConst*
value
B*
dtype0
E
ae/dec4/kernelConst*
dtype0*
valueB@
8
ae/dec4_1/stack/3Const*
valueB *
dtype0
_
ae/dec1_1/MatMulMatMulZ_inae/dec1/kernel*
transpose_a( *
transpose_b( *
T0
\
ae/dec1_1/BiasAddBiasAddae/dec1_1/MatMulae/dec1/bias*
T0*
data_formatNHWC
[
ae/dec1_bn_1/batchnorm/mul_1Mulae/dec1_1/BiasAddae/dec1_bn_1/batchnorm/mul*
T0
f
ae/dec1_bn_1/batchnorm/add_1Addae/dec1_bn_1/batchnorm/mul_1ae/dec1_bn_1/batchnorm/sub*
T0
O
LeakyRelu_3	LeakyReluae/dec1_bn_1/batchnorm/add_1*
alpha%ÍĖL>*
T0
f
ae/dec2_1/MatMulMatMulLeakyRelu_3ae/dec2/kernel*
T0*
transpose_a( *
transpose_b( 
\
ae/dec2_1/BiasAddBiasAddae/dec2_1/MatMulae/dec2/bias*
T0*
data_formatNHWC
[
ae/dec2_bn_1/batchnorm/mul_1Mulae/dec2_1/BiasAddae/dec2_bn_1/batchnorm/mul*
T0
f
ae/dec2_bn_1/batchnorm/add_1Addae/dec2_bn_1/batchnorm/mul_1ae/dec2_bn_1/batchnorm/sub*
T0
O
LeakyRelu_4	LeakyReluae/dec2_bn_1/batchnorm/add_1*
T0*
alpha%ÍĖL>
I
	Reshape_4ReshapeLeakyRelu_4Reshape_4/shape*
T0*
Tshape0
<
ae/dec3_1/ShapeShape	Reshape_4*
T0*
out_type0

ae/dec3_1/strided_slice_2StridedSliceae/dec3_1/Shape!ae/dec4_1/strided_slice_1/stack_1!ae/dec4_1/strided_slice_2/stack_1ae/dec4_1/strided_slice/stack_1*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 

ae/dec3_1/strided_slice_1StridedSliceae/dec3_1/Shapeae/dec4_1/strided_slice/stack_1!ae/dec4_1/strided_slice_1/stack_1ae/dec4_1/strided_slice/stack_1*
end_mask *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 

ae/dec3_1/strided_sliceStridedSliceae/dec3_1/Shapeae/dec4_1/strided_slice/stackae/dec4_1/strided_slice/stack_1ae/dec4_1/strided_slice/stack_1*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
K
ae/dec3_1/mul_1Mulae/dec3_1/strided_slice_2ae/dec4_1/mul/y*
T0
I
ae/dec3_1/mulMulae/dec3_1/strided_slice_1ae/dec4_1/mul/y*
T0

ae/dec3_1/stackPackae/dec3_1/strided_sliceae/dec3_1/mulae/dec3_1/mul_1ae/dec3_1/stack/3*
N*
T0*

axis 
Ô
ae/dec3_1/conv2d_transposeConv2DBackpropInputae/dec3_1/stackae/dec3/kernel	Reshape_4*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
o
ae/dec3_bn_1/FusedBatchNorm/MulMulae/dec3_1/conv2d_transpose"ae/dec3_bn_1/FusedBatchNorm/Scaled*
T0
p
ae/dec3_bn_1/FusedBatchNormAddae/dec3_bn_1/FusedBatchNorm/Mul"ae/dec3_bn_1/FusedBatchNorm/Offset*
T0
N
LeakyRelu_5	LeakyReluae/dec3_bn_1/FusedBatchNorm*
alpha%ÍĖL>*
T0
>
ae/dec4_1/ShapeShapeLeakyRelu_5*
T0*
out_type0

ae/dec4_1/strided_slice_2StridedSliceae/dec4_1/Shape!ae/dec4_1/strided_slice_1/stack_1!ae/dec4_1/strided_slice_2/stack_1ae/dec4_1/strided_slice/stack_1*
end_mask *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 

ae/dec4_1/strided_slice_1StridedSliceae/dec4_1/Shapeae/dec4_1/strided_slice/stack_1!ae/dec4_1/strided_slice_1/stack_1ae/dec4_1/strided_slice/stack_1*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0

ae/dec4_1/strided_sliceStridedSliceae/dec4_1/Shapeae/dec4_1/strided_slice/stackae/dec4_1/strided_slice/stack_1ae/dec4_1/strided_slice/stack_1*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
K
ae/dec4_1/mul_1Mulae/dec4_1/strided_slice_2ae/dec4_1/mul/y*
T0
I
ae/dec4_1/mulMulae/dec4_1/strided_slice_1ae/dec4_1/mul/y*
T0

ae/dec4_1/stackPackae/dec4_1/strided_sliceae/dec4_1/mulae/dec4_1/mul_1ae/dec4_1/stack/3*
T0*

axis *
N
Ö
ae/dec4_1/conv2d_transposeConv2DBackpropInputae/dec4_1/stackae/dec4/kernelLeakyRelu_5*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations

X
	Reshape_5Reshapeae/dec4_1/conv2d_transposeReshape_5/shape*
T0*
Tshape0
$
X_outSigmoid	Reshape_5*
T0 " 