//You should avoid making direct changes.
//Consider using 'partial classes' to extend these types
//Input: caffe.proto

#pragma warning disable CS1591, CS0612, CS3021

namespace CaffemodelLoader
{
    [global::ProtoBuf.ProtoContract()]
    public partial class BlobShape
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"dim", IsPacked = true)]
        public long[] Dims { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class BlobProto
    {
        [global::ProtoBuf.ProtoMember(7, Name = @"shape")]
        public BlobShape Shape { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"data", IsPacked = true)]
        public float[] Datas { get; set; }

        [global::ProtoBuf.ProtoMember(6, Name = @"diff", IsPacked = true)]
        public float[] Diffs { get; set; }

        [global::ProtoBuf.ProtoMember(8, Name = @"double_data", IsPacked = true)]
        public double[] DoubleDatas { get; set; }

        [global::ProtoBuf.ProtoMember(9, Name = @"double_diff", IsPacked = true)]
        public double[] DoubleDiffs { get; set; }

        [global::ProtoBuf.ProtoMember(1, Name = @"num")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int Num
        {
            get { return __pbn__Num ?? 0; }
            set { __pbn__Num = value; }
        }
        public bool ShouldSerializeNum() => __pbn__Num != null;
        public void ResetNum() => __pbn__Num = null;
        private int? __pbn__Num;

        [global::ProtoBuf.ProtoMember(2, Name = @"channels")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int Channels
        {
            get { return __pbn__Channels ?? 0; }
            set { __pbn__Channels = value; }
        }
        public bool ShouldSerializeChannels() => __pbn__Channels != null;
        public void ResetChannels() => __pbn__Channels = null;
        private int? __pbn__Channels;

        [global::ProtoBuf.ProtoMember(3, Name = @"height")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int Height
        {
            get { return __pbn__Height ?? 0; }
            set { __pbn__Height = value; }
        }
        public bool ShouldSerializeHeight() => __pbn__Height != null;
        public void ResetHeight() => __pbn__Height = null;
        private int? __pbn__Height;

        [global::ProtoBuf.ProtoMember(4, Name = @"width")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int Width
        {
            get { return __pbn__Width ?? 0; }
            set { __pbn__Width = value; }
        }
        public bool ShouldSerializeWidth() => __pbn__Width != null;
        public void ResetWidth() => __pbn__Width = null;
        private int? __pbn__Width;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class BlobProtoVector
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"blobs")]
        public global::System.Collections.Generic.List<BlobProto> Blobs { get; } = new global::System.Collections.Generic.List<BlobProto>();

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class Datum
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"channels")]
        public int Channels
        {
            get { return __pbn__Channels.GetValueOrDefault(); }
            set { __pbn__Channels = value; }
        }
        public bool ShouldSerializeChannels() => __pbn__Channels != null;
        public void ResetChannels() => __pbn__Channels = null;
        private int? __pbn__Channels;

        [global::ProtoBuf.ProtoMember(2, Name = @"height")]
        public int Height
        {
            get { return __pbn__Height.GetValueOrDefault(); }
            set { __pbn__Height = value; }
        }
        public bool ShouldSerializeHeight() => __pbn__Height != null;
        public void ResetHeight() => __pbn__Height = null;
        private int? __pbn__Height;

        [global::ProtoBuf.ProtoMember(3, Name = @"width")]
        public int Width
        {
            get { return __pbn__Width.GetValueOrDefault(); }
            set { __pbn__Width = value; }
        }
        public bool ShouldSerializeWidth() => __pbn__Width != null;
        public void ResetWidth() => __pbn__Width = null;
        private int? __pbn__Width;

        [global::ProtoBuf.ProtoMember(4, Name = @"data")]
        public byte[] Data
        {
            get { return __pbn__Data; }
            set { __pbn__Data = value; }
        }
        public bool ShouldSerializeData() => __pbn__Data != null;
        public void ResetData() => __pbn__Data = null;
        private byte[] __pbn__Data;

        [global::ProtoBuf.ProtoMember(5, Name = @"label")]
        public int Label
        {
            get { return __pbn__Label.GetValueOrDefault(); }
            set { __pbn__Label = value; }
        }
        public bool ShouldSerializeLabel() => __pbn__Label != null;
        public void ResetLabel() => __pbn__Label = null;
        private int? __pbn__Label;

        [global::ProtoBuf.ProtoMember(6, Name = @"float_data")]
        public float[] FloatDatas { get; set; }

        [global::ProtoBuf.ProtoMember(7, Name = @"encoded")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool Encoded
        {
            get { return __pbn__Encoded ?? false; }
            set { __pbn__Encoded = value; }
        }
        public bool ShouldSerializeEncoded() => __pbn__Encoded != null;
        public void ResetEncoded() => __pbn__Encoded = null;
        private bool? __pbn__Encoded;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class FillerParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"type")]
        [global::System.ComponentModel.DefaultValue(@"constant")]
        public string Type
        {
            get { return __pbn__Type ?? @"constant"; }
            set { __pbn__Type = value; }
        }
        public bool ShouldSerializeType() => __pbn__Type != null;
        public void ResetType() => __pbn__Type = null;
        private string __pbn__Type;

        [global::ProtoBuf.ProtoMember(2, Name = @"value")]
        [global::System.ComponentModel.DefaultValue(0)]
        public float Value
        {
            get { return __pbn__Value ?? 0; }
            set { __pbn__Value = value; }
        }
        public bool ShouldSerializeValue() => __pbn__Value != null;
        public void ResetValue() => __pbn__Value = null;
        private float? __pbn__Value;

        [global::ProtoBuf.ProtoMember(3, Name = @"min")]
        [global::System.ComponentModel.DefaultValue(0)]
        public float Min
        {
            get { return __pbn__Min ?? 0; }
            set { __pbn__Min = value; }
        }
        public bool ShouldSerializeMin() => __pbn__Min != null;
        public void ResetMin() => __pbn__Min = null;
        private float? __pbn__Min;

        [global::ProtoBuf.ProtoMember(4, Name = @"max")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Max
        {
            get { return __pbn__Max ?? 1; }
            set { __pbn__Max = value; }
        }
        public bool ShouldSerializeMax() => __pbn__Max != null;
        public void ResetMax() => __pbn__Max = null;
        private float? __pbn__Max;

        [global::ProtoBuf.ProtoMember(5, Name = @"mean")]
        [global::System.ComponentModel.DefaultValue(0)]
        public float Mean
        {
            get { return __pbn__Mean ?? 0; }
            set { __pbn__Mean = value; }
        }
        public bool ShouldSerializeMean() => __pbn__Mean != null;
        public void ResetMean() => __pbn__Mean = null;
        private float? __pbn__Mean;

        [global::ProtoBuf.ProtoMember(6, Name = @"std")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Std
        {
            get { return __pbn__Std ?? 1; }
            set { __pbn__Std = value; }
        }
        public bool ShouldSerializeStd() => __pbn__Std != null;
        public void ResetStd() => __pbn__Std = null;
        private float? __pbn__Std;

        [global::ProtoBuf.ProtoMember(7, Name = @"sparse")]
        [global::System.ComponentModel.DefaultValue(-1)]
        public int Sparse
        {
            get { return __pbn__Sparse ?? -1; }
            set { __pbn__Sparse = value; }
        }
        public bool ShouldSerializeSparse() => __pbn__Sparse != null;
        public void ResetSparse() => __pbn__Sparse = null;
        private int? __pbn__Sparse;

        [global::ProtoBuf.ProtoMember(8)]
        [global::System.ComponentModel.DefaultValue(VarianceNorm.FanIn)]
        public VarianceNorm variance_norm
        {
            get { return __pbn__variance_norm ?? VarianceNorm.FanIn; }
            set { __pbn__variance_norm = value; }
        }
        public bool ShouldSerializevariance_norm() => __pbn__variance_norm != null;
        public void Resetvariance_norm() => __pbn__variance_norm = null;
        private VarianceNorm? __pbn__variance_norm;

        [global::ProtoBuf.ProtoContract()]
        public enum VarianceNorm
        {
            [global::ProtoBuf.ProtoEnum(Name = @"FAN_IN")]
            FanIn = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"FAN_OUT")]
            FanOut = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"AVERAGE")]
            Average = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class NetParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Name
        {
            get { return __pbn__Name ?? ""; }
            set { __pbn__Name = value; }
        }
        public bool ShouldSerializeName() => __pbn__Name != null;
        public void ResetName() => __pbn__Name = null;
        private string __pbn__Name;

        [global::ProtoBuf.ProtoMember(3, Name = @"input")]
        public global::System.Collections.Generic.List<string> Inputs { get; } = new global::System.Collections.Generic.List<string>();

        [global::ProtoBuf.ProtoMember(8, Name = @"input_shape")]
        public global::System.Collections.Generic.List<BlobShape> InputShapes { get; } = new global::System.Collections.Generic.List<BlobShape>();

        [global::ProtoBuf.ProtoMember(4, Name = @"input_dim")]
        public int[] InputDims { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"force_backward")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool ForceBackward
        {
            get { return __pbn__ForceBackward ?? false; }
            set { __pbn__ForceBackward = value; }
        }
        public bool ShouldSerializeForceBackward() => __pbn__ForceBackward != null;
        public void ResetForceBackward() => __pbn__ForceBackward = null;
        private bool? __pbn__ForceBackward;

        [global::ProtoBuf.ProtoMember(6, Name = @"state")]
        public NetState State { get; set; }

        [global::ProtoBuf.ProtoMember(7, Name = @"debug_info")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool DebugInfo
        {
            get { return __pbn__DebugInfo ?? false; }
            set { __pbn__DebugInfo = value; }
        }
        public bool ShouldSerializeDebugInfo() => __pbn__DebugInfo != null;
        public void ResetDebugInfo() => __pbn__DebugInfo = null;
        private bool? __pbn__DebugInfo;

        [global::ProtoBuf.ProtoMember(100, Name = @"layer")]
        public global::System.Collections.Generic.List<LayerParameter> Layer { get; } = new global::System.Collections.Generic.List<LayerParameter>();

        [global::ProtoBuf.ProtoMember(2, Name = @"layers")]
        public global::System.Collections.Generic.List<V1LayerParameter> Layers { get; } = new global::System.Collections.Generic.List<V1LayerParameter>();

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class SolverParameter
    {
        [global::ProtoBuf.ProtoMember(24, Name = @"net")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Net
        {
            get { return __pbn__Net ?? ""; }
            set { __pbn__Net = value; }
        }
        public bool ShouldSerializeNet() => __pbn__Net != null;
        public void ResetNet() => __pbn__Net = null;
        private string __pbn__Net;

        [global::ProtoBuf.ProtoMember(25, Name = @"net_param")]
        public NetParameter NetParam { get; set; }

        [global::ProtoBuf.ProtoMember(1, Name = @"train_net")]
        [global::System.ComponentModel.DefaultValue("")]
        public string TrainNet
        {
            get { return __pbn__TrainNet ?? ""; }
            set { __pbn__TrainNet = value; }
        }
        public bool ShouldSerializeTrainNet() => __pbn__TrainNet != null;
        public void ResetTrainNet() => __pbn__TrainNet = null;
        private string __pbn__TrainNet;

        [global::ProtoBuf.ProtoMember(2, Name = @"test_net")]
        public global::System.Collections.Generic.List<string> TestNets { get; } = new global::System.Collections.Generic.List<string>();

        [global::ProtoBuf.ProtoMember(21, Name = @"train_net_param")]
        public NetParameter TrainNetParam { get; set; }

        [global::ProtoBuf.ProtoMember(22, Name = @"test_net_param")]
        public global::System.Collections.Generic.List<NetParameter> TestNetParams { get; } = new global::System.Collections.Generic.List<NetParameter>();

        [global::ProtoBuf.ProtoMember(26, Name = @"train_state")]
        public NetState TrainState { get; set; }

        [global::ProtoBuf.ProtoMember(27, Name = @"test_state")]
        public global::System.Collections.Generic.List<NetState> TestStates { get; } = new global::System.Collections.Generic.List<NetState>();

        [global::ProtoBuf.ProtoMember(3, Name = @"test_iter")]
        public int[] TestIters { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"test_interval")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int TestInterval
        {
            get { return __pbn__TestInterval ?? 0; }
            set { __pbn__TestInterval = value; }
        }
        public bool ShouldSerializeTestInterval() => __pbn__TestInterval != null;
        public void ResetTestInterval() => __pbn__TestInterval = null;
        private int? __pbn__TestInterval;

        [global::ProtoBuf.ProtoMember(19, Name = @"test_compute_loss")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool TestComputeLoss
        {
            get { return __pbn__TestComputeLoss ?? false; }
            set { __pbn__TestComputeLoss = value; }
        }
        public bool ShouldSerializeTestComputeLoss() => __pbn__TestComputeLoss != null;
        public void ResetTestComputeLoss() => __pbn__TestComputeLoss = null;
        private bool? __pbn__TestComputeLoss;

        [global::ProtoBuf.ProtoMember(32, Name = @"test_initialization")]
        [global::System.ComponentModel.DefaultValue(true)]
        public bool TestInitialization
        {
            get { return __pbn__TestInitialization ?? true; }
            set { __pbn__TestInitialization = value; }
        }
        public bool ShouldSerializeTestInitialization() => __pbn__TestInitialization != null;
        public void ResetTestInitialization() => __pbn__TestInitialization = null;
        private bool? __pbn__TestInitialization;

        [global::ProtoBuf.ProtoMember(5, Name = @"base_lr")]
        public float BaseLr
        {
            get { return __pbn__BaseLr.GetValueOrDefault(); }
            set { __pbn__BaseLr = value; }
        }
        public bool ShouldSerializeBaseLr() => __pbn__BaseLr != null;
        public void ResetBaseLr() => __pbn__BaseLr = null;
        private float? __pbn__BaseLr;

        [global::ProtoBuf.ProtoMember(6, Name = @"display")]
        public int Display
        {
            get { return __pbn__Display.GetValueOrDefault(); }
            set { __pbn__Display = value; }
        }
        public bool ShouldSerializeDisplay() => __pbn__Display != null;
        public void ResetDisplay() => __pbn__Display = null;
        private int? __pbn__Display;

        [global::ProtoBuf.ProtoMember(33, Name = @"average_loss")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int AverageLoss
        {
            get { return __pbn__AverageLoss ?? 1; }
            set { __pbn__AverageLoss = value; }
        }
        public bool ShouldSerializeAverageLoss() => __pbn__AverageLoss != null;
        public void ResetAverageLoss() => __pbn__AverageLoss = null;
        private int? __pbn__AverageLoss;

        [global::ProtoBuf.ProtoMember(7, Name = @"max_iter")]
        public int MaxIter
        {
            get { return __pbn__MaxIter.GetValueOrDefault(); }
            set { __pbn__MaxIter = value; }
        }
        public bool ShouldSerializeMaxIter() => __pbn__MaxIter != null;
        public void ResetMaxIter() => __pbn__MaxIter = null;
        private int? __pbn__MaxIter;

        [global::ProtoBuf.ProtoMember(36, Name = @"iter_size")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int IterSize
        {
            get { return __pbn__IterSize ?? 1; }
            set { __pbn__IterSize = value; }
        }
        public bool ShouldSerializeIterSize() => __pbn__IterSize != null;
        public void ResetIterSize() => __pbn__IterSize = null;
        private int? __pbn__IterSize;

        [global::ProtoBuf.ProtoMember(8, Name = @"lr_policy")]
        [global::System.ComponentModel.DefaultValue("")]
        public string LrPolicy
        {
            get { return __pbn__LrPolicy ?? ""; }
            set { __pbn__LrPolicy = value; }
        }
        public bool ShouldSerializeLrPolicy() => __pbn__LrPolicy != null;
        public void ResetLrPolicy() => __pbn__LrPolicy = null;
        private string __pbn__LrPolicy;

        [global::ProtoBuf.ProtoMember(9, Name = @"gamma")]
        public float Gamma
        {
            get { return __pbn__Gamma.GetValueOrDefault(); }
            set { __pbn__Gamma = value; }
        }
        public bool ShouldSerializeGamma() => __pbn__Gamma != null;
        public void ResetGamma() => __pbn__Gamma = null;
        private float? __pbn__Gamma;

        [global::ProtoBuf.ProtoMember(10, Name = @"power")]
        public float Power
        {
            get { return __pbn__Power.GetValueOrDefault(); }
            set { __pbn__Power = value; }
        }
        public bool ShouldSerializePower() => __pbn__Power != null;
        public void ResetPower() => __pbn__Power = null;
        private float? __pbn__Power;

        [global::ProtoBuf.ProtoMember(11, Name = @"momentum")]
        public float Momentum
        {
            get { return __pbn__Momentum.GetValueOrDefault(); }
            set { __pbn__Momentum = value; }
        }
        public bool ShouldSerializeMomentum() => __pbn__Momentum != null;
        public void ResetMomentum() => __pbn__Momentum = null;
        private float? __pbn__Momentum;

        [global::ProtoBuf.ProtoMember(12, Name = @"weight_decay")]
        public float WeightDecay
        {
            get { return __pbn__WeightDecay.GetValueOrDefault(); }
            set { __pbn__WeightDecay = value; }
        }
        public bool ShouldSerializeWeightDecay() => __pbn__WeightDecay != null;
        public void ResetWeightDecay() => __pbn__WeightDecay = null;
        private float? __pbn__WeightDecay;

        [global::ProtoBuf.ProtoMember(29, Name = @"regularization_type")]
        [global::System.ComponentModel.DefaultValue(@"L2")]
        public string RegularizationType
        {
            get { return __pbn__RegularizationType ?? @"L2"; }
            set { __pbn__RegularizationType = value; }
        }
        public bool ShouldSerializeRegularizationType() => __pbn__RegularizationType != null;
        public void ResetRegularizationType() => __pbn__RegularizationType = null;
        private string __pbn__RegularizationType;

        [global::ProtoBuf.ProtoMember(13, Name = @"stepsize")]
        public int Stepsize
        {
            get { return __pbn__Stepsize.GetValueOrDefault(); }
            set { __pbn__Stepsize = value; }
        }
        public bool ShouldSerializeStepsize() => __pbn__Stepsize != null;
        public void ResetStepsize() => __pbn__Stepsize = null;
        private int? __pbn__Stepsize;

        [global::ProtoBuf.ProtoMember(34, Name = @"stepvalue")]
        public int[] Stepvalues { get; set; }

        [global::ProtoBuf.ProtoMember(35, Name = @"clip_gradients")]
        [global::System.ComponentModel.DefaultValue(-1)]
        public float ClipGradients
        {
            get { return __pbn__ClipGradients ?? -1; }
            set { __pbn__ClipGradients = value; }
        }
        public bool ShouldSerializeClipGradients() => __pbn__ClipGradients != null;
        public void ResetClipGradients() => __pbn__ClipGradients = null;
        private float? __pbn__ClipGradients;

        [global::ProtoBuf.ProtoMember(14, Name = @"snapshot")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int Snapshot
        {
            get { return __pbn__Snapshot ?? 0; }
            set { __pbn__Snapshot = value; }
        }
        public bool ShouldSerializeSnapshot() => __pbn__Snapshot != null;
        public void ResetSnapshot() => __pbn__Snapshot = null;
        private int? __pbn__Snapshot;

        [global::ProtoBuf.ProtoMember(15, Name = @"snapshot_prefix")]
        [global::System.ComponentModel.DefaultValue("")]
        public string SnapshotPrefix
        {
            get { return __pbn__SnapshotPrefix ?? ""; }
            set { __pbn__SnapshotPrefix = value; }
        }
        public bool ShouldSerializeSnapshotPrefix() => __pbn__SnapshotPrefix != null;
        public void ResetSnapshotPrefix() => __pbn__SnapshotPrefix = null;
        private string __pbn__SnapshotPrefix;

        [global::ProtoBuf.ProtoMember(16, Name = @"snapshot_diff")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool SnapshotDiff
        {
            get { return __pbn__SnapshotDiff ?? false; }
            set { __pbn__SnapshotDiff = value; }
        }
        public bool ShouldSerializeSnapshotDiff() => __pbn__SnapshotDiff != null;
        public void ResetSnapshotDiff() => __pbn__SnapshotDiff = null;
        private bool? __pbn__SnapshotDiff;

        [global::ProtoBuf.ProtoMember(37)]
        [global::System.ComponentModel.DefaultValue(SnapshotFormat.Binaryproto)]
        public SnapshotFormat snapshot_format
        {
            get { return __pbn__snapshot_format ?? SnapshotFormat.Binaryproto; }
            set { __pbn__snapshot_format = value; }
        }
        public bool ShouldSerializesnapshot_format() => __pbn__snapshot_format != null;
        public void Resetsnapshot_format() => __pbn__snapshot_format = null;
        private SnapshotFormat? __pbn__snapshot_format;

        [global::ProtoBuf.ProtoMember(17)]
        [global::System.ComponentModel.DefaultValue(SolverMode.Gpu)]
        public SolverMode solver_mode
        {
            get { return __pbn__solver_mode ?? SolverMode.Gpu; }
            set { __pbn__solver_mode = value; }
        }
        public bool ShouldSerializesolver_mode() => __pbn__solver_mode != null;
        public void Resetsolver_mode() => __pbn__solver_mode = null;
        private SolverMode? __pbn__solver_mode;

        [global::ProtoBuf.ProtoMember(18, Name = @"device_id")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int DeviceId
        {
            get { return __pbn__DeviceId ?? 0; }
            set { __pbn__DeviceId = value; }
        }
        public bool ShouldSerializeDeviceId() => __pbn__DeviceId != null;
        public void ResetDeviceId() => __pbn__DeviceId = null;
        private int? __pbn__DeviceId;

        [global::ProtoBuf.ProtoMember(20, Name = @"random_seed")]
        [global::System.ComponentModel.DefaultValue(-1)]
        public long RandomSeed
        {
            get { return __pbn__RandomSeed ?? -1; }
            set { __pbn__RandomSeed = value; }
        }
        public bool ShouldSerializeRandomSeed() => __pbn__RandomSeed != null;
        public void ResetRandomSeed() => __pbn__RandomSeed = null;
        private long? __pbn__RandomSeed;

        [global::ProtoBuf.ProtoMember(40, Name = @"type")]
        [global::System.ComponentModel.DefaultValue(@"SGD")]
        public string Type
        {
            get { return __pbn__Type ?? @"SGD"; }
            set { __pbn__Type = value; }
        }
        public bool ShouldSerializeType() => __pbn__Type != null;
        public void ResetType() => __pbn__Type = null;
        private string __pbn__Type;

        [global::ProtoBuf.ProtoMember(31, Name = @"delta")]
        [global::System.ComponentModel.DefaultValue(1e-008f)]
        public float Delta
        {
            get { return __pbn__Delta ?? 1e-008f; }
            set { __pbn__Delta = value; }
        }
        public bool ShouldSerializeDelta() => __pbn__Delta != null;
        public void ResetDelta() => __pbn__Delta = null;
        private float? __pbn__Delta;

        [global::ProtoBuf.ProtoMember(39, Name = @"momentum2")]
        [global::System.ComponentModel.DefaultValue(0.999f)]
        public float Momentum2
        {
            get { return __pbn__Momentum2 ?? 0.999f; }
            set { __pbn__Momentum2 = value; }
        }
        public bool ShouldSerializeMomentum2() => __pbn__Momentum2 != null;
        public void ResetMomentum2() => __pbn__Momentum2 = null;
        private float? __pbn__Momentum2;

        [global::ProtoBuf.ProtoMember(38, Name = @"rms_decay")]
        [global::System.ComponentModel.DefaultValue(0.99f)]
        public float RmsDecay
        {
            get { return __pbn__RmsDecay ?? 0.99f; }
            set { __pbn__RmsDecay = value; }
        }
        public bool ShouldSerializeRmsDecay() => __pbn__RmsDecay != null;
        public void ResetRmsDecay() => __pbn__RmsDecay = null;
        private float? __pbn__RmsDecay;

        [global::ProtoBuf.ProtoMember(23, Name = @"debug_info")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool DebugInfo
        {
            get { return __pbn__DebugInfo ?? false; }
            set { __pbn__DebugInfo = value; }
        }
        public bool ShouldSerializeDebugInfo() => __pbn__DebugInfo != null;
        public void ResetDebugInfo() => __pbn__DebugInfo = null;
        private bool? __pbn__DebugInfo;

        [global::ProtoBuf.ProtoMember(28, Name = @"snapshot_after_train")]
        [global::System.ComponentModel.DefaultValue(true)]
        public bool SnapshotAfterTrain
        {
            get { return __pbn__SnapshotAfterTrain ?? true; }
            set { __pbn__SnapshotAfterTrain = value; }
        }
        public bool ShouldSerializeSnapshotAfterTrain() => __pbn__SnapshotAfterTrain != null;
        public void ResetSnapshotAfterTrain() => __pbn__SnapshotAfterTrain = null;
        private bool? __pbn__SnapshotAfterTrain;

        [global::ProtoBuf.ProtoMember(30)]
        [global::System.ComponentModel.DefaultValue(SolverType.Sgd)]
        public SolverType solver_type
        {
            get { return __pbn__solver_type ?? SolverType.Sgd; }
            set { __pbn__solver_type = value; }
        }
        public bool ShouldSerializesolver_type() => __pbn__solver_type != null;
        public void Resetsolver_type() => __pbn__solver_type = null;
        private SolverType? __pbn__solver_type;

        [global::ProtoBuf.ProtoMember(41, Name = @"layer_wise_reduce")]
        [global::System.ComponentModel.DefaultValue(true)]
        public bool LayerWiseReduce
        {
            get { return __pbn__LayerWiseReduce ?? true; }
            set { __pbn__LayerWiseReduce = value; }
        }
        public bool ShouldSerializeLayerWiseReduce() => __pbn__LayerWiseReduce != null;
        public void ResetLayerWiseReduce() => __pbn__LayerWiseReduce = null;
        private bool? __pbn__LayerWiseReduce;

        [global::ProtoBuf.ProtoContract()]
        public enum SnapshotFormat
        {
            [global::ProtoBuf.ProtoEnum(Name = @"HDF5")]
            Hdf5 = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"BINARYPROTO")]
            Binaryproto = 1,
        }

        [global::ProtoBuf.ProtoContract()]
        public enum SolverMode
        {
            [global::ProtoBuf.ProtoEnum(Name = @"CPU")]
            Cpu = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"GPU")]
            Gpu = 1,
        }

        [global::ProtoBuf.ProtoContract()]
        public enum SolverType
        {
            [global::ProtoBuf.ProtoEnum(Name = @"SGD")]
            Sgd = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"NESTEROV")]
            Nesterov = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"ADAGRAD")]
            Adagrad = 2,
            [global::ProtoBuf.ProtoEnum(Name = @"RMSPROP")]
            Rmsprop = 3,
            [global::ProtoBuf.ProtoEnum(Name = @"ADADELTA")]
            Adadelta = 4,
            [global::ProtoBuf.ProtoEnum(Name = @"ADAM")]
            Adam = 5,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class SolverState
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"iter")]
        public int Iter
        {
            get { return __pbn__Iter.GetValueOrDefault(); }
            set { __pbn__Iter = value; }
        }
        public bool ShouldSerializeIter() => __pbn__Iter != null;
        public void ResetIter() => __pbn__Iter = null;
        private int? __pbn__Iter;

        [global::ProtoBuf.ProtoMember(2, Name = @"learned_net")]
        [global::System.ComponentModel.DefaultValue("")]
        public string LearnedNet
        {
            get { return __pbn__LearnedNet ?? ""; }
            set { __pbn__LearnedNet = value; }
        }
        public bool ShouldSerializeLearnedNet() => __pbn__LearnedNet != null;
        public void ResetLearnedNet() => __pbn__LearnedNet = null;
        private string __pbn__LearnedNet;

        [global::ProtoBuf.ProtoMember(3, Name = @"history")]
        public global::System.Collections.Generic.List<BlobProto> Histories { get; } = new global::System.Collections.Generic.List<BlobProto>();

        [global::ProtoBuf.ProtoMember(4, Name = @"current_step")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int CurrentStep
        {
            get { return __pbn__CurrentStep ?? 0; }
            set { __pbn__CurrentStep = value; }
        }
        public bool ShouldSerializeCurrentStep() => __pbn__CurrentStep != null;
        public void ResetCurrentStep() => __pbn__CurrentStep = null;
        private int? __pbn__CurrentStep;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class NetState
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"phase")]
        [global::System.ComponentModel.DefaultValue(Phase.Test)]
        public Phase Phase
        {
            get { return __pbn__Phase ?? Phase.Test; }
            set { __pbn__Phase = value; }
        }
        public bool ShouldSerializePhase() => __pbn__Phase != null;
        public void ResetPhase() => __pbn__Phase = null;
        private Phase? __pbn__Phase;

        [global::ProtoBuf.ProtoMember(2, Name = @"level")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int Level
        {
            get { return __pbn__Level ?? 0; }
            set { __pbn__Level = value; }
        }
        public bool ShouldSerializeLevel() => __pbn__Level != null;
        public void ResetLevel() => __pbn__Level = null;
        private int? __pbn__Level;

        [global::ProtoBuf.ProtoMember(3, Name = @"stage")]
        public global::System.Collections.Generic.List<string> Stages { get; } = new global::System.Collections.Generic.List<string>();

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class NetStateRule
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"phase")]
        [global::System.ComponentModel.DefaultValue(Phase.Train)]
        public Phase Phase
        {
            get { return __pbn__Phase ?? Phase.Train; }
            set { __pbn__Phase = value; }
        }
        public bool ShouldSerializePhase() => __pbn__Phase != null;
        public void ResetPhase() => __pbn__Phase = null;
        private Phase? __pbn__Phase;

        [global::ProtoBuf.ProtoMember(2, Name = @"min_level")]
        public int MinLevel
        {
            get { return __pbn__MinLevel.GetValueOrDefault(); }
            set { __pbn__MinLevel = value; }
        }
        public bool ShouldSerializeMinLevel() => __pbn__MinLevel != null;
        public void ResetMinLevel() => __pbn__MinLevel = null;
        private int? __pbn__MinLevel;

        [global::ProtoBuf.ProtoMember(3, Name = @"max_level")]
        public int MaxLevel
        {
            get { return __pbn__MaxLevel.GetValueOrDefault(); }
            set { __pbn__MaxLevel = value; }
        }
        public bool ShouldSerializeMaxLevel() => __pbn__MaxLevel != null;
        public void ResetMaxLevel() => __pbn__MaxLevel = null;
        private int? __pbn__MaxLevel;

        [global::ProtoBuf.ProtoMember(4, Name = @"stage")]
        public global::System.Collections.Generic.List<string> Stages { get; } = new global::System.Collections.Generic.List<string>();

        [global::ProtoBuf.ProtoMember(5, Name = @"not_stage")]
        public global::System.Collections.Generic.List<string> NotStages { get; } = new global::System.Collections.Generic.List<string>();

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ParamSpec
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Name
        {
            get { return __pbn__Name ?? ""; }
            set { __pbn__Name = value; }
        }
        public bool ShouldSerializeName() => __pbn__Name != null;
        public void ResetName() => __pbn__Name = null;
        private string __pbn__Name;

        [global::ProtoBuf.ProtoMember(2, Name = @"share_mode")]
        [global::System.ComponentModel.DefaultValue(DimCheckMode.Strict)]
        public DimCheckMode ShareMode
        {
            get { return __pbn__ShareMode ?? DimCheckMode.Strict; }
            set { __pbn__ShareMode = value; }
        }
        public bool ShouldSerializeShareMode() => __pbn__ShareMode != null;
        public void ResetShareMode() => __pbn__ShareMode = null;
        private DimCheckMode? __pbn__ShareMode;

        [global::ProtoBuf.ProtoMember(3, Name = @"lr_mult")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float LrMult
        {
            get { return __pbn__LrMult ?? 1; }
            set { __pbn__LrMult = value; }
        }
        public bool ShouldSerializeLrMult() => __pbn__LrMult != null;
        public void ResetLrMult() => __pbn__LrMult = null;
        private float? __pbn__LrMult;

        [global::ProtoBuf.ProtoMember(4, Name = @"decay_mult")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float DecayMult
        {
            get { return __pbn__DecayMult ?? 1; }
            set { __pbn__DecayMult = value; }
        }
        public bool ShouldSerializeDecayMult() => __pbn__DecayMult != null;
        public void ResetDecayMult() => __pbn__DecayMult = null;
        private float? __pbn__DecayMult;

        [global::ProtoBuf.ProtoContract()]
        public enum DimCheckMode
        {
            [global::ProtoBuf.ProtoEnum(Name = @"STRICT")]
            Strict = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"PERMISSIVE")]
            Permissive = 1,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class LayerParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Name
        {
            get { return __pbn__Name ?? ""; }
            set { __pbn__Name = value; }
        }
        public bool ShouldSerializeName() => __pbn__Name != null;
        public void ResetName() => __pbn__Name = null;
        private string __pbn__Name;

        [global::ProtoBuf.ProtoMember(2, Name = @"type")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Type
        {
            get { return __pbn__Type ?? ""; }
            set { __pbn__Type = value; }
        }
        public bool ShouldSerializeType() => __pbn__Type != null;
        public void ResetType() => __pbn__Type = null;
        private string __pbn__Type;

        [global::ProtoBuf.ProtoMember(3, Name = @"bottom")]
        public global::System.Collections.Generic.List<string> Bottoms { get; } = new global::System.Collections.Generic.List<string>();

        [global::ProtoBuf.ProtoMember(4, Name = @"top")]
        public global::System.Collections.Generic.List<string> Tops { get; } = new global::System.Collections.Generic.List<string>();

        [global::ProtoBuf.ProtoMember(10, Name = @"phase")]
        [global::System.ComponentModel.DefaultValue(Phase.Train)]
        public Phase Phase
        {
            get { return __pbn__Phase ?? Phase.Train; }
            set { __pbn__Phase = value; }
        }
        public bool ShouldSerializePhase() => __pbn__Phase != null;
        public void ResetPhase() => __pbn__Phase = null;
        private Phase? __pbn__Phase;

        [global::ProtoBuf.ProtoMember(5, Name = @"loss_weight")]
        public float[] LossWeights { get; set; }

        [global::ProtoBuf.ProtoMember(6, Name = @"param")]
        public global::System.Collections.Generic.List<ParamSpec> Params { get; } = new global::System.Collections.Generic.List<ParamSpec>();

        [global::ProtoBuf.ProtoMember(7, Name = @"blobs")]
        public global::System.Collections.Generic.List<BlobProto> Blobs { get; } = new global::System.Collections.Generic.List<BlobProto>();

        [global::ProtoBuf.ProtoMember(11, Name = @"propagate_down")]
        public bool[] PropagateDowns { get; set; }

        [global::ProtoBuf.ProtoMember(8, Name = @"include")]
        public global::System.Collections.Generic.List<NetStateRule> Includes { get; } = new global::System.Collections.Generic.List<NetStateRule>();

        [global::ProtoBuf.ProtoMember(9, Name = @"exclude")]
        public global::System.Collections.Generic.List<NetStateRule> Excludes { get; } = new global::System.Collections.Generic.List<NetStateRule>();

        [global::ProtoBuf.ProtoMember(100, Name = @"transform_param")]
        public TransformationParameter TransformParam { get; set; }

        [global::ProtoBuf.ProtoMember(101, Name = @"loss_param")]
        public LossParameter LossParam { get; set; }

        [global::ProtoBuf.ProtoMember(102, Name = @"accuracy_param")]
        public AccuracyParameter AccuracyParam { get; set; }

        [global::ProtoBuf.ProtoMember(103, Name = @"argmax_param")]
        public ArgMaxParameter ArgmaxParam { get; set; }

        [global::ProtoBuf.ProtoMember(139, Name = @"batch_norm_param")]
        public BatchNormParameter BatchNormParam { get; set; }

        [global::ProtoBuf.ProtoMember(141, Name = @"bias_param")]
        public BiasParameter BiasParam { get; set; }

        [global::ProtoBuf.ProtoMember(104, Name = @"concat_param")]
        public ConcatParameter ConcatParam { get; set; }

        [global::ProtoBuf.ProtoMember(105, Name = @"contrastive_loss_param")]
        public ContrastiveLossParameter ContrastiveLossParam { get; set; }

        [global::ProtoBuf.ProtoMember(106, Name = @"convolution_param")]
        public ConvolutionParameter ConvolutionParam { get; set; }

        [global::ProtoBuf.ProtoMember(144, Name = @"crop_param")]
        public CropParameter CropParam { get; set; }

        [global::ProtoBuf.ProtoMember(107, Name = @"data_param")]
        public DataParameter DataParam { get; set; }

        [global::ProtoBuf.ProtoMember(108, Name = @"dropout_param")]
        public DropoutParameter DropoutParam { get; set; }

        [global::ProtoBuf.ProtoMember(109, Name = @"dummy_data_param")]
        public DummyDataParameter DummyDataParam { get; set; }

        [global::ProtoBuf.ProtoMember(110, Name = @"eltwise_param")]
        public EltwiseParameter EltwiseParam { get; set; }

        [global::ProtoBuf.ProtoMember(140, Name = @"elu_param")]
        public ELUParameter EluParam { get; set; }

        [global::ProtoBuf.ProtoMember(137, Name = @"embed_param")]
        public EmbedParameter EmbedParam { get; set; }

        [global::ProtoBuf.ProtoMember(111, Name = @"exp_param")]
        public ExpParameter ExpParam { get; set; }

        [global::ProtoBuf.ProtoMember(135, Name = @"flatten_param")]
        public FlattenParameter FlattenParam { get; set; }

        [global::ProtoBuf.ProtoMember(112, Name = @"hdf5_data_param")]
        public HDF5DataParameter Hdf5DataParam { get; set; }

        [global::ProtoBuf.ProtoMember(113, Name = @"hdf5_output_param")]
        public HDF5OutputParameter Hdf5OutputParam { get; set; }

        [global::ProtoBuf.ProtoMember(114, Name = @"hinge_loss_param")]
        public HingeLossParameter HingeLossParam { get; set; }

        [global::ProtoBuf.ProtoMember(115, Name = @"image_data_param")]
        public ImageDataParameter ImageDataParam { get; set; }

        [global::ProtoBuf.ProtoMember(116, Name = @"infogain_loss_param")]
        public InfogainLossParameter InfogainLossParam { get; set; }

        [global::ProtoBuf.ProtoMember(117, Name = @"inner_product_param")]
        public InnerProductParameter InnerProductParam { get; set; }

        [global::ProtoBuf.ProtoMember(143, Name = @"input_param")]
        public InputParameter InputParam { get; set; }

        [global::ProtoBuf.ProtoMember(134, Name = @"log_param")]
        public LogParameter LogParam { get; set; }

        [global::ProtoBuf.ProtoMember(118, Name = @"lrn_param")]
        public LRNParameter LrnParam { get; set; }

        [global::ProtoBuf.ProtoMember(119, Name = @"memory_data_param")]
        public MemoryDataParameter MemoryDataParam { get; set; }

        [global::ProtoBuf.ProtoMember(120, Name = @"mvn_param")]
        public MVNParameter MvnParam { get; set; }

        [global::ProtoBuf.ProtoMember(145, Name = @"parameter_param")]
        public ParameterParameter ParameterParam { get; set; }

        [global::ProtoBuf.ProtoMember(121, Name = @"pooling_param")]
        public PoolingParameter PoolingParam { get; set; }

        [global::ProtoBuf.ProtoMember(122, Name = @"power_param")]
        public PowerParameter PowerParam { get; set; }

        [global::ProtoBuf.ProtoMember(131, Name = @"prelu_param")]
        public PReLUParameter PreluParam { get; set; }

        [global::ProtoBuf.ProtoMember(130, Name = @"python_param")]
        public PythonParameter PythonParam { get; set; }

        [global::ProtoBuf.ProtoMember(146, Name = @"recurrent_param")]
        public RecurrentParameter RecurrentParam { get; set; }

        [global::ProtoBuf.ProtoMember(136, Name = @"reduction_param")]
        public ReductionParameter ReductionParam { get; set; }

        [global::ProtoBuf.ProtoMember(123, Name = @"relu_param")]
        public ReLUParameter ReluParam { get; set; }

        [global::ProtoBuf.ProtoMember(133, Name = @"reshape_param")]
        public ReshapeParameter ReshapeParam { get; set; }

        [global::ProtoBuf.ProtoMember(142, Name = @"scale_param")]
        public ScaleParameter ScaleParam { get; set; }

        [global::ProtoBuf.ProtoMember(124, Name = @"sigmoid_param")]
        public SigmoidParameter SigmoidParam { get; set; }

        [global::ProtoBuf.ProtoMember(125, Name = @"softmax_param")]
        public SoftmaxParameter SoftmaxParam { get; set; }

        [global::ProtoBuf.ProtoMember(132, Name = @"spp_param")]
        public SPPParameter SppParam { get; set; }

        [global::ProtoBuf.ProtoMember(126, Name = @"slice_param")]
        public SliceParameter SliceParam { get; set; }

        [global::ProtoBuf.ProtoMember(127, Name = @"tanh_param")]
        public TanHParameter TanhParam { get; set; }

        [global::ProtoBuf.ProtoMember(128, Name = @"threshold_param")]
        public ThresholdParameter ThresholdParam { get; set; }

        [global::ProtoBuf.ProtoMember(138, Name = @"tile_param")]
        public TileParameter TileParam { get; set; }

        [global::ProtoBuf.ProtoMember(129, Name = @"window_data_param")]
        public WindowDataParameter WindowDataParam { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class TransformationParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"scale")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Scale
        {
            get { return __pbn__Scale ?? 1; }
            set { __pbn__Scale = value; }
        }
        public bool ShouldSerializeScale() => __pbn__Scale != null;
        public void ResetScale() => __pbn__Scale = null;
        private float? __pbn__Scale;

        [global::ProtoBuf.ProtoMember(2, Name = @"mirror")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool Mirror
        {
            get { return __pbn__Mirror ?? false; }
            set { __pbn__Mirror = value; }
        }
        public bool ShouldSerializeMirror() => __pbn__Mirror != null;
        public void ResetMirror() => __pbn__Mirror = null;
        private bool? __pbn__Mirror;

        [global::ProtoBuf.ProtoMember(3, Name = @"crop_size")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint CropSize
        {
            get { return __pbn__CropSize ?? 0; }
            set { __pbn__CropSize = value; }
        }
        public bool ShouldSerializeCropSize() => __pbn__CropSize != null;
        public void ResetCropSize() => __pbn__CropSize = null;
        private uint? __pbn__CropSize;

        [global::ProtoBuf.ProtoMember(4, Name = @"mean_file")]
        [global::System.ComponentModel.DefaultValue("")]
        public string MeanFile
        {
            get { return __pbn__MeanFile ?? ""; }
            set { __pbn__MeanFile = value; }
        }
        public bool ShouldSerializeMeanFile() => __pbn__MeanFile != null;
        public void ResetMeanFile() => __pbn__MeanFile = null;
        private string __pbn__MeanFile;

        [global::ProtoBuf.ProtoMember(5, Name = @"mean_value")]
        public float[] MeanValues { get; set; }

        [global::ProtoBuf.ProtoMember(6, Name = @"force_color")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool ForceColor
        {
            get { return __pbn__ForceColor ?? false; }
            set { __pbn__ForceColor = value; }
        }
        public bool ShouldSerializeForceColor() => __pbn__ForceColor != null;
        public void ResetForceColor() => __pbn__ForceColor = null;
        private bool? __pbn__ForceColor;

        [global::ProtoBuf.ProtoMember(7, Name = @"force_gray")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool ForceGray
        {
            get { return __pbn__ForceGray ?? false; }
            set { __pbn__ForceGray = value; }
        }
        public bool ShouldSerializeForceGray() => __pbn__ForceGray != null;
        public void ResetForceGray() => __pbn__ForceGray = null;
        private bool? __pbn__ForceGray;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class LossParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"ignore_label")]
        public int IgnoreLabel
        {
            get { return __pbn__IgnoreLabel.GetValueOrDefault(); }
            set { __pbn__IgnoreLabel = value; }
        }
        public bool ShouldSerializeIgnoreLabel() => __pbn__IgnoreLabel != null;
        public void ResetIgnoreLabel() => __pbn__IgnoreLabel = null;
        private int? __pbn__IgnoreLabel;

        [global::ProtoBuf.ProtoMember(3, Name = @"normalization")]
        [global::System.ComponentModel.DefaultValue(NormalizationMode.Valid)]
        public NormalizationMode Normalization
        {
            get { return __pbn__Normalization ?? NormalizationMode.Valid; }
            set { __pbn__Normalization = value; }
        }
        public bool ShouldSerializeNormalization() => __pbn__Normalization != null;
        public void ResetNormalization() => __pbn__Normalization = null;
        private NormalizationMode? __pbn__Normalization;

        [global::ProtoBuf.ProtoMember(2, Name = @"normalize")]
        public bool Normalize
        {
            get { return __pbn__Normalize.GetValueOrDefault(); }
            set { __pbn__Normalize = value; }
        }
        public bool ShouldSerializeNormalize() => __pbn__Normalize != null;
        public void ResetNormalize() => __pbn__Normalize = null;
        private bool? __pbn__Normalize;

        [global::ProtoBuf.ProtoContract()]
        public enum NormalizationMode
        {
            [global::ProtoBuf.ProtoEnum(Name = @"FULL")]
            Full = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"VALID")]
            Valid = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"BATCH_SIZE")]
            BatchSize = 2,
            [global::ProtoBuf.ProtoEnum(Name = @"NONE")]
            None = 3,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class AccuracyParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"top_k")]
        [global::System.ComponentModel.DefaultValue(1)]
        public uint TopK
        {
            get { return __pbn__TopK ?? 1; }
            set { __pbn__TopK = value; }
        }
        public bool ShouldSerializeTopK() => __pbn__TopK != null;
        public void ResetTopK() => __pbn__TopK = null;
        private uint? __pbn__TopK;

        [global::ProtoBuf.ProtoMember(2, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(3, Name = @"ignore_label")]
        public int IgnoreLabel
        {
            get { return __pbn__IgnoreLabel.GetValueOrDefault(); }
            set { __pbn__IgnoreLabel = value; }
        }
        public bool ShouldSerializeIgnoreLabel() => __pbn__IgnoreLabel != null;
        public void ResetIgnoreLabel() => __pbn__IgnoreLabel = null;
        private int? __pbn__IgnoreLabel;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ArgMaxParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"out_max_val")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool OutMaxVal
        {
            get { return __pbn__OutMaxVal ?? false; }
            set { __pbn__OutMaxVal = value; }
        }
        public bool ShouldSerializeOutMaxVal() => __pbn__OutMaxVal != null;
        public void ResetOutMaxVal() => __pbn__OutMaxVal = null;
        private bool? __pbn__OutMaxVal;

        [global::ProtoBuf.ProtoMember(2, Name = @"top_k")]
        [global::System.ComponentModel.DefaultValue(1)]
        public uint TopK
        {
            get { return __pbn__TopK ?? 1; }
            set { __pbn__TopK = value; }
        }
        public bool ShouldSerializeTopK() => __pbn__TopK != null;
        public void ResetTopK() => __pbn__TopK = null;
        private uint? __pbn__TopK;

        [global::ProtoBuf.ProtoMember(3, Name = @"axis")]
        public int Axis
        {
            get { return __pbn__Axis.GetValueOrDefault(); }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ConcatParameter
    {
        [global::ProtoBuf.ProtoMember(2, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(1, Name = @"concat_dim")]
        [global::System.ComponentModel.DefaultValue(1)]
        public uint ConcatDim
        {
            get { return __pbn__ConcatDim ?? 1; }
            set { __pbn__ConcatDim = value; }
        }
        public bool ShouldSerializeConcatDim() => __pbn__ConcatDim != null;
        public void ResetConcatDim() => __pbn__ConcatDim = null;
        private uint? __pbn__ConcatDim;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class BatchNormParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"use_global_stats")]
        public bool UseGlobalStats
        {
            get { return __pbn__UseGlobalStats.GetValueOrDefault(); }
            set { __pbn__UseGlobalStats = value; }
        }
        public bool ShouldSerializeUseGlobalStats() => __pbn__UseGlobalStats != null;
        public void ResetUseGlobalStats() => __pbn__UseGlobalStats = null;
        private bool? __pbn__UseGlobalStats;

        [global::ProtoBuf.ProtoMember(2, Name = @"moving_average_fraction")]
        [global::System.ComponentModel.DefaultValue(0.999f)]
        public float MovingAverageFraction
        {
            get { return __pbn__MovingAverageFraction ?? 0.999f; }
            set { __pbn__MovingAverageFraction = value; }
        }
        public bool ShouldSerializeMovingAverageFraction() => __pbn__MovingAverageFraction != null;
        public void ResetMovingAverageFraction() => __pbn__MovingAverageFraction = null;
        private float? __pbn__MovingAverageFraction;

        [global::ProtoBuf.ProtoMember(3, Name = @"eps")]
        [global::System.ComponentModel.DefaultValue(1e-005f)]
        public float Eps
        {
            get { return __pbn__Eps ?? 1e-005f; }
            set { __pbn__Eps = value; }
        }
        public bool ShouldSerializeEps() => __pbn__Eps != null;
        public void ResetEps() => __pbn__Eps = null;
        private float? __pbn__Eps;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class BiasParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(2, Name = @"num_axes")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int NumAxes
        {
            get { return __pbn__NumAxes ?? 1; }
            set { __pbn__NumAxes = value; }
        }
        public bool ShouldSerializeNumAxes() => __pbn__NumAxes != null;
        public void ResetNumAxes() => __pbn__NumAxes = null;
        private int? __pbn__NumAxes;

        [global::ProtoBuf.ProtoMember(3, Name = @"filler")]
        public FillerParameter Filler { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ContrastiveLossParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"margin")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Margin
        {
            get { return __pbn__Margin ?? 1; }
            set { __pbn__Margin = value; }
        }
        public bool ShouldSerializeMargin() => __pbn__Margin != null;
        public void ResetMargin() => __pbn__Margin = null;
        private float? __pbn__Margin;

        [global::ProtoBuf.ProtoMember(2, Name = @"legacy_version")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool LegacyVersion
        {
            get { return __pbn__LegacyVersion ?? false; }
            set { __pbn__LegacyVersion = value; }
        }
        public bool ShouldSerializeLegacyVersion() => __pbn__LegacyVersion != null;
        public void ResetLegacyVersion() => __pbn__LegacyVersion = null;
        private bool? __pbn__LegacyVersion;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ConvolutionParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"num_output")]
        public uint NumOutput
        {
            get { return __pbn__NumOutput.GetValueOrDefault(); }
            set { __pbn__NumOutput = value; }
        }
        public bool ShouldSerializeNumOutput() => __pbn__NumOutput != null;
        public void ResetNumOutput() => __pbn__NumOutput = null;
        private uint? __pbn__NumOutput;

        [global::ProtoBuf.ProtoMember(2, Name = @"bias_term")]
        [global::System.ComponentModel.DefaultValue(true)]
        public bool BiasTerm
        {
            get { return __pbn__BiasTerm ?? true; }
            set { __pbn__BiasTerm = value; }
        }
        public bool ShouldSerializeBiasTerm() => __pbn__BiasTerm != null;
        public void ResetBiasTerm() => __pbn__BiasTerm = null;
        private bool? __pbn__BiasTerm;

        [global::ProtoBuf.ProtoMember(3, Name = @"pad")]
        public uint[] Pads { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"kernel_size")]
        public uint[] KernelSizes { get; set; }

        [global::ProtoBuf.ProtoMember(6, Name = @"stride")]
        public uint[] Strides { get; set; }

        [global::ProtoBuf.ProtoMember(18, Name = @"dilation")]
        public uint[] Dilations { get; set; }

        [global::ProtoBuf.ProtoMember(9, Name = @"pad_h")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint PadH
        {
            get { return __pbn__PadH ?? 0; }
            set { __pbn__PadH = value; }
        }
        public bool ShouldSerializePadH() => __pbn__PadH != null;
        public void ResetPadH() => __pbn__PadH = null;
        private uint? __pbn__PadH;

        [global::ProtoBuf.ProtoMember(10, Name = @"pad_w")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint PadW
        {
            get { return __pbn__PadW ?? 0; }
            set { __pbn__PadW = value; }
        }
        public bool ShouldSerializePadW() => __pbn__PadW != null;
        public void ResetPadW() => __pbn__PadW = null;
        private uint? __pbn__PadW;

        [global::ProtoBuf.ProtoMember(11, Name = @"kernel_h")]
        public uint KernelH
        {
            get { return __pbn__KernelH.GetValueOrDefault(); }
            set { __pbn__KernelH = value; }
        }
        public bool ShouldSerializeKernelH() => __pbn__KernelH != null;
        public void ResetKernelH() => __pbn__KernelH = null;
        private uint? __pbn__KernelH;

        [global::ProtoBuf.ProtoMember(12, Name = @"kernel_w")]
        public uint KernelW
        {
            get { return __pbn__KernelW.GetValueOrDefault(); }
            set { __pbn__KernelW = value; }
        }
        public bool ShouldSerializeKernelW() => __pbn__KernelW != null;
        public void ResetKernelW() => __pbn__KernelW = null;
        private uint? __pbn__KernelW;

        [global::ProtoBuf.ProtoMember(13, Name = @"stride_h")]
        public uint StrideH
        {
            get { return __pbn__StrideH.GetValueOrDefault(); }
            set { __pbn__StrideH = value; }
        }
        public bool ShouldSerializeStrideH() => __pbn__StrideH != null;
        public void ResetStrideH() => __pbn__StrideH = null;
        private uint? __pbn__StrideH;

        [global::ProtoBuf.ProtoMember(14, Name = @"stride_w")]
        public uint StrideW
        {
            get { return __pbn__StrideW.GetValueOrDefault(); }
            set { __pbn__StrideW = value; }
        }
        public bool ShouldSerializeStrideW() => __pbn__StrideW != null;
        public void ResetStrideW() => __pbn__StrideW = null;
        private uint? __pbn__StrideW;

        [global::ProtoBuf.ProtoMember(5, Name = @"group")]
        [global::System.ComponentModel.DefaultValue(1)]
        public uint Group
        {
            get { return __pbn__Group ?? 1; }
            set { __pbn__Group = value; }
        }
        public bool ShouldSerializeGroup() => __pbn__Group != null;
        public void ResetGroup() => __pbn__Group = null;
        private uint? __pbn__Group;

        [global::ProtoBuf.ProtoMember(7, Name = @"weight_filler")]
        public FillerParameter WeightFiller { get; set; }

        [global::ProtoBuf.ProtoMember(8, Name = @"bias_filler")]
        public FillerParameter BiasFiller { get; set; }

        [global::ProtoBuf.ProtoMember(15)]
        [global::System.ComponentModel.DefaultValue(Engine.Default)]
        public Engine engine
        {
            get { return __pbn__engine ?? Engine.Default; }
            set { __pbn__engine = value; }
        }
        public bool ShouldSerializeengine() => __pbn__engine != null;
        public void Resetengine() => __pbn__engine = null;
        private Engine? __pbn__engine;

        [global::ProtoBuf.ProtoMember(16, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(17, Name = @"force_nd_im2col")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool ForceNdIm2col
        {
            get { return __pbn__ForceNdIm2col ?? false; }
            set { __pbn__ForceNdIm2col = value; }
        }
        public bool ShouldSerializeForceNdIm2col() => __pbn__ForceNdIm2col != null;
        public void ResetForceNdIm2col() => __pbn__ForceNdIm2col = null;
        private bool? __pbn__ForceNdIm2col;

        [global::ProtoBuf.ProtoContract()]
        public enum Engine
        {
            [global::ProtoBuf.ProtoEnum(Name = @"DEFAULT")]
            Default = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"CAFFE")]
            Caffe = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"CUDNN")]
            Cudnn = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class CropParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(2)]
        public int Axis
        {
            get { return __pbn__Axis ?? 2; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(2, Name = @"offset")]
        public uint[] Offsets { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class DataParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"source")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Source
        {
            get { return __pbn__Source ?? ""; }
            set { __pbn__Source = value; }
        }
        public bool ShouldSerializeSource() => __pbn__Source != null;
        public void ResetSource() => __pbn__Source = null;
        private string __pbn__Source;

        [global::ProtoBuf.ProtoMember(4, Name = @"batch_size")]
        public uint BatchSize
        {
            get { return __pbn__BatchSize.GetValueOrDefault(); }
            set { __pbn__BatchSize = value; }
        }
        public bool ShouldSerializeBatchSize() => __pbn__BatchSize != null;
        public void ResetBatchSize() => __pbn__BatchSize = null;
        private uint? __pbn__BatchSize;

        [global::ProtoBuf.ProtoMember(7, Name = @"rand_skip")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint RandSkip
        {
            get { return __pbn__RandSkip ?? 0; }
            set { __pbn__RandSkip = value; }
        }
        public bool ShouldSerializeRandSkip() => __pbn__RandSkip != null;
        public void ResetRandSkip() => __pbn__RandSkip = null;
        private uint? __pbn__RandSkip;

        [global::ProtoBuf.ProtoMember(8, Name = @"backend")]
        [global::System.ComponentModel.DefaultValue(Db.Leveldb)]
        public Db Backend
        {
            get { return __pbn__Backend ?? Db.Leveldb; }
            set { __pbn__Backend = value; }
        }
        public bool ShouldSerializeBackend() => __pbn__Backend != null;
        public void ResetBackend() => __pbn__Backend = null;
        private Db? __pbn__Backend;

        [global::ProtoBuf.ProtoMember(2, Name = @"scale")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Scale
        {
            get { return __pbn__Scale ?? 1; }
            set { __pbn__Scale = value; }
        }
        public bool ShouldSerializeScale() => __pbn__Scale != null;
        public void ResetScale() => __pbn__Scale = null;
        private float? __pbn__Scale;

        [global::ProtoBuf.ProtoMember(3, Name = @"mean_file")]
        [global::System.ComponentModel.DefaultValue("")]
        public string MeanFile
        {
            get { return __pbn__MeanFile ?? ""; }
            set { __pbn__MeanFile = value; }
        }
        public bool ShouldSerializeMeanFile() => __pbn__MeanFile != null;
        public void ResetMeanFile() => __pbn__MeanFile = null;
        private string __pbn__MeanFile;

        [global::ProtoBuf.ProtoMember(5, Name = @"crop_size")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint CropSize
        {
            get { return __pbn__CropSize ?? 0; }
            set { __pbn__CropSize = value; }
        }
        public bool ShouldSerializeCropSize() => __pbn__CropSize != null;
        public void ResetCropSize() => __pbn__CropSize = null;
        private uint? __pbn__CropSize;

        [global::ProtoBuf.ProtoMember(6, Name = @"mirror")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool Mirror
        {
            get { return __pbn__Mirror ?? false; }
            set { __pbn__Mirror = value; }
        }
        public bool ShouldSerializeMirror() => __pbn__Mirror != null;
        public void ResetMirror() => __pbn__Mirror = null;
        private bool? __pbn__Mirror;

        [global::ProtoBuf.ProtoMember(9, Name = @"force_encoded_color")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool ForceEncodedColor
        {
            get { return __pbn__ForceEncodedColor ?? false; }
            set { __pbn__ForceEncodedColor = value; }
        }
        public bool ShouldSerializeForceEncodedColor() => __pbn__ForceEncodedColor != null;
        public void ResetForceEncodedColor() => __pbn__ForceEncodedColor = null;
        private bool? __pbn__ForceEncodedColor;

        [global::ProtoBuf.ProtoMember(10, Name = @"prefetch")]
        [global::System.ComponentModel.DefaultValue(4)]
        public uint Prefetch
        {
            get { return __pbn__Prefetch ?? 4; }
            set { __pbn__Prefetch = value; }
        }
        public bool ShouldSerializePrefetch() => __pbn__Prefetch != null;
        public void ResetPrefetch() => __pbn__Prefetch = null;
        private uint? __pbn__Prefetch;

        [global::ProtoBuf.ProtoContract(Name = @"DB")]
        public enum Db
        {
            [global::ProtoBuf.ProtoEnum(Name = @"LEVELDB")]
            Leveldb = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"LMDB")]
            Lmdb = 1,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class DropoutParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"dropout_ratio")]
        [global::System.ComponentModel.DefaultValue(0.5f)]
        public float DropoutRatio
        {
            get { return __pbn__DropoutRatio ?? 0.5f; }
            set { __pbn__DropoutRatio = value; }
        }
        public bool ShouldSerializeDropoutRatio() => __pbn__DropoutRatio != null;
        public void ResetDropoutRatio() => __pbn__DropoutRatio = null;
        private float? __pbn__DropoutRatio;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class DummyDataParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"data_filler")]
        public global::System.Collections.Generic.List<FillerParameter> DataFillers { get; } = new global::System.Collections.Generic.List<FillerParameter>();

        [global::ProtoBuf.ProtoMember(6, Name = @"shape")]
        public global::System.Collections.Generic.List<BlobShape> Shapes { get; } = new global::System.Collections.Generic.List<BlobShape>();

        [global::ProtoBuf.ProtoMember(2, Name = @"num")]
        public uint[] Nums { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"channels")]
        public uint[] Channels { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"height")]
        public uint[] Heights { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"width")]
        public uint[] Widths { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class EltwiseParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"operation")]
        [global::System.ComponentModel.DefaultValue(EltwiseOp.Sum)]
        public EltwiseOp Operation
        {
            get { return __pbn__Operation ?? EltwiseOp.Sum; }
            set { __pbn__Operation = value; }
        }
        public bool ShouldSerializeOperation() => __pbn__Operation != null;
        public void ResetOperation() => __pbn__Operation = null;
        private EltwiseOp? __pbn__Operation;

        [global::ProtoBuf.ProtoMember(2, Name = @"coeff")]
        public float[] Coeffs { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"stable_prod_grad")]
        [global::System.ComponentModel.DefaultValue(true)]
        public bool StableProdGrad
        {
            get { return __pbn__StableProdGrad ?? true; }
            set { __pbn__StableProdGrad = value; }
        }
        public bool ShouldSerializeStableProdGrad() => __pbn__StableProdGrad != null;
        public void ResetStableProdGrad() => __pbn__StableProdGrad = null;
        private bool? __pbn__StableProdGrad;

        [global::ProtoBuf.ProtoContract()]
        public enum EltwiseOp
        {
            [global::ProtoBuf.ProtoEnum(Name = @"PROD")]
            Prod = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"SUM")]
            Sum = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"MAX")]
            Max = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ELUParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"alpha")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Alpha
        {
            get { return __pbn__Alpha ?? 1; }
            set { __pbn__Alpha = value; }
        }
        public bool ShouldSerializeAlpha() => __pbn__Alpha != null;
        public void ResetAlpha() => __pbn__Alpha = null;
        private float? __pbn__Alpha;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class EmbedParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"num_output")]
        public uint NumOutput
        {
            get { return __pbn__NumOutput.GetValueOrDefault(); }
            set { __pbn__NumOutput = value; }
        }
        public bool ShouldSerializeNumOutput() => __pbn__NumOutput != null;
        public void ResetNumOutput() => __pbn__NumOutput = null;
        private uint? __pbn__NumOutput;

        [global::ProtoBuf.ProtoMember(2, Name = @"input_dim")]
        public uint InputDim
        {
            get { return __pbn__InputDim.GetValueOrDefault(); }
            set { __pbn__InputDim = value; }
        }
        public bool ShouldSerializeInputDim() => __pbn__InputDim != null;
        public void ResetInputDim() => __pbn__InputDim = null;
        private uint? __pbn__InputDim;

        [global::ProtoBuf.ProtoMember(3, Name = @"bias_term")]
        [global::System.ComponentModel.DefaultValue(true)]
        public bool BiasTerm
        {
            get { return __pbn__BiasTerm ?? true; }
            set { __pbn__BiasTerm = value; }
        }
        public bool ShouldSerializeBiasTerm() => __pbn__BiasTerm != null;
        public void ResetBiasTerm() => __pbn__BiasTerm = null;
        private bool? __pbn__BiasTerm;

        [global::ProtoBuf.ProtoMember(4, Name = @"weight_filler")]
        public FillerParameter WeightFiller { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"bias_filler")]
        public FillerParameter BiasFiller { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ExpParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"base")]
        [global::System.ComponentModel.DefaultValue(-1)]
        public float Base
        {
            get { return __pbn__Base ?? -1; }
            set { __pbn__Base = value; }
        }
        public bool ShouldSerializeBase() => __pbn__Base != null;
        public void ResetBase() => __pbn__Base = null;
        private float? __pbn__Base;

        [global::ProtoBuf.ProtoMember(2, Name = @"scale")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Scale
        {
            get { return __pbn__Scale ?? 1; }
            set { __pbn__Scale = value; }
        }
        public bool ShouldSerializeScale() => __pbn__Scale != null;
        public void ResetScale() => __pbn__Scale = null;
        private float? __pbn__Scale;

        [global::ProtoBuf.ProtoMember(3, Name = @"shift")]
        [global::System.ComponentModel.DefaultValue(0)]
        public float Shift
        {
            get { return __pbn__Shift ?? 0; }
            set { __pbn__Shift = value; }
        }
        public bool ShouldSerializeShift() => __pbn__Shift != null;
        public void ResetShift() => __pbn__Shift = null;
        private float? __pbn__Shift;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class FlattenParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(2, Name = @"end_axis")]
        [global::System.ComponentModel.DefaultValue(-1)]
        public int EndAxis
        {
            get { return __pbn__EndAxis ?? -1; }
            set { __pbn__EndAxis = value; }
        }
        public bool ShouldSerializeEndAxis() => __pbn__EndAxis != null;
        public void ResetEndAxis() => __pbn__EndAxis = null;
        private int? __pbn__EndAxis;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class HDF5DataParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"source")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Source
        {
            get { return __pbn__Source ?? ""; }
            set { __pbn__Source = value; }
        }
        public bool ShouldSerializeSource() => __pbn__Source != null;
        public void ResetSource() => __pbn__Source = null;
        private string __pbn__Source;

        [global::ProtoBuf.ProtoMember(2, Name = @"batch_size")]
        public uint BatchSize
        {
            get { return __pbn__BatchSize.GetValueOrDefault(); }
            set { __pbn__BatchSize = value; }
        }
        public bool ShouldSerializeBatchSize() => __pbn__BatchSize != null;
        public void ResetBatchSize() => __pbn__BatchSize = null;
        private uint? __pbn__BatchSize;

        [global::ProtoBuf.ProtoMember(3, Name = @"shuffle")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool Shuffle
        {
            get { return __pbn__Shuffle ?? false; }
            set { __pbn__Shuffle = value; }
        }
        public bool ShouldSerializeShuffle() => __pbn__Shuffle != null;
        public void ResetShuffle() => __pbn__Shuffle = null;
        private bool? __pbn__Shuffle;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class HDF5OutputParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"file_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string FileName
        {
            get { return __pbn__FileName ?? ""; }
            set { __pbn__FileName = value; }
        }
        public bool ShouldSerializeFileName() => __pbn__FileName != null;
        public void ResetFileName() => __pbn__FileName = null;
        private string __pbn__FileName;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class HingeLossParameter
    {
        [global::ProtoBuf.ProtoMember(1)]
        [global::System.ComponentModel.DefaultValue(Norm.L1)]
        public Norm norm
        {
            get { return __pbn__norm ?? Norm.L1; }
            set { __pbn__norm = value; }
        }
        public bool ShouldSerializenorm() => __pbn__norm != null;
        public void Resetnorm() => __pbn__norm = null;
        private Norm? __pbn__norm;

        [global::ProtoBuf.ProtoContract()]
        public enum Norm
        {
            L1 = 1,
            L2 = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ImageDataParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"source")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Source
        {
            get { return __pbn__Source ?? ""; }
            set { __pbn__Source = value; }
        }
        public bool ShouldSerializeSource() => __pbn__Source != null;
        public void ResetSource() => __pbn__Source = null;
        private string __pbn__Source;

        [global::ProtoBuf.ProtoMember(4, Name = @"batch_size")]
        [global::System.ComponentModel.DefaultValue(1)]
        public uint BatchSize
        {
            get { return __pbn__BatchSize ?? 1; }
            set { __pbn__BatchSize = value; }
        }
        public bool ShouldSerializeBatchSize() => __pbn__BatchSize != null;
        public void ResetBatchSize() => __pbn__BatchSize = null;
        private uint? __pbn__BatchSize;

        [global::ProtoBuf.ProtoMember(7, Name = @"rand_skip")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint RandSkip
        {
            get { return __pbn__RandSkip ?? 0; }
            set { __pbn__RandSkip = value; }
        }
        public bool ShouldSerializeRandSkip() => __pbn__RandSkip != null;
        public void ResetRandSkip() => __pbn__RandSkip = null;
        private uint? __pbn__RandSkip;

        [global::ProtoBuf.ProtoMember(8, Name = @"shuffle")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool Shuffle
        {
            get { return __pbn__Shuffle ?? false; }
            set { __pbn__Shuffle = value; }
        }
        public bool ShouldSerializeShuffle() => __pbn__Shuffle != null;
        public void ResetShuffle() => __pbn__Shuffle = null;
        private bool? __pbn__Shuffle;

        [global::ProtoBuf.ProtoMember(9, Name = @"new_height")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint NewHeight
        {
            get { return __pbn__NewHeight ?? 0; }
            set { __pbn__NewHeight = value; }
        }
        public bool ShouldSerializeNewHeight() => __pbn__NewHeight != null;
        public void ResetNewHeight() => __pbn__NewHeight = null;
        private uint? __pbn__NewHeight;

        [global::ProtoBuf.ProtoMember(10, Name = @"new_width")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint NewWidth
        {
            get { return __pbn__NewWidth ?? 0; }
            set { __pbn__NewWidth = value; }
        }
        public bool ShouldSerializeNewWidth() => __pbn__NewWidth != null;
        public void ResetNewWidth() => __pbn__NewWidth = null;
        private uint? __pbn__NewWidth;

        [global::ProtoBuf.ProtoMember(11, Name = @"is_color")]
        [global::System.ComponentModel.DefaultValue(true)]
        public bool IsColor
        {
            get { return __pbn__IsColor ?? true; }
            set { __pbn__IsColor = value; }
        }
        public bool ShouldSerializeIsColor() => __pbn__IsColor != null;
        public void ResetIsColor() => __pbn__IsColor = null;
        private bool? __pbn__IsColor;

        [global::ProtoBuf.ProtoMember(2, Name = @"scale")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Scale
        {
            get { return __pbn__Scale ?? 1; }
            set { __pbn__Scale = value; }
        }
        public bool ShouldSerializeScale() => __pbn__Scale != null;
        public void ResetScale() => __pbn__Scale = null;
        private float? __pbn__Scale;

        [global::ProtoBuf.ProtoMember(3, Name = @"mean_file")]
        [global::System.ComponentModel.DefaultValue("")]
        public string MeanFile
        {
            get { return __pbn__MeanFile ?? ""; }
            set { __pbn__MeanFile = value; }
        }
        public bool ShouldSerializeMeanFile() => __pbn__MeanFile != null;
        public void ResetMeanFile() => __pbn__MeanFile = null;
        private string __pbn__MeanFile;

        [global::ProtoBuf.ProtoMember(5, Name = @"crop_size")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint CropSize
        {
            get { return __pbn__CropSize ?? 0; }
            set { __pbn__CropSize = value; }
        }
        public bool ShouldSerializeCropSize() => __pbn__CropSize != null;
        public void ResetCropSize() => __pbn__CropSize = null;
        private uint? __pbn__CropSize;

        [global::ProtoBuf.ProtoMember(6, Name = @"mirror")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool Mirror
        {
            get { return __pbn__Mirror ?? false; }
            set { __pbn__Mirror = value; }
        }
        public bool ShouldSerializeMirror() => __pbn__Mirror != null;
        public void ResetMirror() => __pbn__Mirror = null;
        private bool? __pbn__Mirror;

        [global::ProtoBuf.ProtoMember(12, Name = @"root_folder")]
        [global::System.ComponentModel.DefaultValue("")]
        public string RootFolder
        {
            get { return __pbn__RootFolder ?? ""; }
            set { __pbn__RootFolder = value; }
        }
        public bool ShouldSerializeRootFolder() => __pbn__RootFolder != null;
        public void ResetRootFolder() => __pbn__RootFolder = null;
        private string __pbn__RootFolder;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class InfogainLossParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"source")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Source
        {
            get { return __pbn__Source ?? ""; }
            set { __pbn__Source = value; }
        }
        public bool ShouldSerializeSource() => __pbn__Source != null;
        public void ResetSource() => __pbn__Source = null;
        private string __pbn__Source;

        [global::ProtoBuf.ProtoMember(2, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class InnerProductParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"num_output")]
        public uint NumOutput
        {
            get { return __pbn__NumOutput.GetValueOrDefault(); }
            set { __pbn__NumOutput = value; }
        }
        public bool ShouldSerializeNumOutput() => __pbn__NumOutput != null;
        public void ResetNumOutput() => __pbn__NumOutput = null;
        private uint? __pbn__NumOutput;

        [global::ProtoBuf.ProtoMember(2, Name = @"bias_term")]
        [global::System.ComponentModel.DefaultValue(true)]
        public bool BiasTerm
        {
            get { return __pbn__BiasTerm ?? true; }
            set { __pbn__BiasTerm = value; }
        }
        public bool ShouldSerializeBiasTerm() => __pbn__BiasTerm != null;
        public void ResetBiasTerm() => __pbn__BiasTerm = null;
        private bool? __pbn__BiasTerm;

        [global::ProtoBuf.ProtoMember(3, Name = @"weight_filler")]
        public FillerParameter WeightFiller { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"bias_filler")]
        public FillerParameter BiasFiller { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(6, Name = @"transpose")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool Transpose
        {
            get { return __pbn__Transpose ?? false; }
            set { __pbn__Transpose = value; }
        }
        public bool ShouldSerializeTranspose() => __pbn__Transpose != null;
        public void ResetTranspose() => __pbn__Transpose = null;
        private bool? __pbn__Transpose;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class InputParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"shape")]
        public global::System.Collections.Generic.List<BlobShape> Shapes { get; } = new global::System.Collections.Generic.List<BlobShape>();

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class LogParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"base")]
        [global::System.ComponentModel.DefaultValue(-1)]
        public float Base
        {
            get { return __pbn__Base ?? -1; }
            set { __pbn__Base = value; }
        }
        public bool ShouldSerializeBase() => __pbn__Base != null;
        public void ResetBase() => __pbn__Base = null;
        private float? __pbn__Base;

        [global::ProtoBuf.ProtoMember(2, Name = @"scale")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Scale
        {
            get { return __pbn__Scale ?? 1; }
            set { __pbn__Scale = value; }
        }
        public bool ShouldSerializeScale() => __pbn__Scale != null;
        public void ResetScale() => __pbn__Scale = null;
        private float? __pbn__Scale;

        [global::ProtoBuf.ProtoMember(3, Name = @"shift")]
        [global::System.ComponentModel.DefaultValue(0)]
        public float Shift
        {
            get { return __pbn__Shift ?? 0; }
            set { __pbn__Shift = value; }
        }
        public bool ShouldSerializeShift() => __pbn__Shift != null;
        public void ResetShift() => __pbn__Shift = null;
        private float? __pbn__Shift;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class LRNParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"local_size")]
        [global::System.ComponentModel.DefaultValue(5)]
        public uint LocalSize
        {
            get { return __pbn__LocalSize ?? 5; }
            set { __pbn__LocalSize = value; }
        }
        public bool ShouldSerializeLocalSize() => __pbn__LocalSize != null;
        public void ResetLocalSize() => __pbn__LocalSize = null;
        private uint? __pbn__LocalSize;

        [global::ProtoBuf.ProtoMember(2, Name = @"alpha")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Alpha
        {
            get { return __pbn__Alpha ?? 1; }
            set { __pbn__Alpha = value; }
        }
        public bool ShouldSerializeAlpha() => __pbn__Alpha != null;
        public void ResetAlpha() => __pbn__Alpha = null;
        private float? __pbn__Alpha;

        [global::ProtoBuf.ProtoMember(3, Name = @"beta")]
        [global::System.ComponentModel.DefaultValue(0.75f)]
        public float Beta
        {
            get { return __pbn__Beta ?? 0.75f; }
            set { __pbn__Beta = value; }
        }
        public bool ShouldSerializeBeta() => __pbn__Beta != null;
        public void ResetBeta() => __pbn__Beta = null;
        private float? __pbn__Beta;

        [global::ProtoBuf.ProtoMember(4)]
        [global::System.ComponentModel.DefaultValue(NormRegion.AcrossChannels)]
        public NormRegion norm_region
        {
            get { return __pbn__norm_region ?? NormRegion.AcrossChannels; }
            set { __pbn__norm_region = value; }
        }
        public bool ShouldSerializenorm_region() => __pbn__norm_region != null;
        public void Resetnorm_region() => __pbn__norm_region = null;
        private NormRegion? __pbn__norm_region;

        [global::ProtoBuf.ProtoMember(5, Name = @"k")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float K
        {
            get { return __pbn__K ?? 1; }
            set { __pbn__K = value; }
        }
        public bool ShouldSerializeK() => __pbn__K != null;
        public void ResetK() => __pbn__K = null;
        private float? __pbn__K;

        [global::ProtoBuf.ProtoMember(6)]
        [global::System.ComponentModel.DefaultValue(Engine.Default)]
        public Engine engine
        {
            get { return __pbn__engine ?? Engine.Default; }
            set { __pbn__engine = value; }
        }
        public bool ShouldSerializeengine() => __pbn__engine != null;
        public void Resetengine() => __pbn__engine = null;
        private Engine? __pbn__engine;

        [global::ProtoBuf.ProtoContract()]
        public enum NormRegion
        {
            [global::ProtoBuf.ProtoEnum(Name = @"ACROSS_CHANNELS")]
            AcrossChannels = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"WITHIN_CHANNEL")]
            WithinChannel = 1,
        }

        [global::ProtoBuf.ProtoContract()]
        public enum Engine
        {
            [global::ProtoBuf.ProtoEnum(Name = @"DEFAULT")]
            Default = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"CAFFE")]
            Caffe = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"CUDNN")]
            Cudnn = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class MemoryDataParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"batch_size")]
        public uint BatchSize
        {
            get { return __pbn__BatchSize.GetValueOrDefault(); }
            set { __pbn__BatchSize = value; }
        }
        public bool ShouldSerializeBatchSize() => __pbn__BatchSize != null;
        public void ResetBatchSize() => __pbn__BatchSize = null;
        private uint? __pbn__BatchSize;

        [global::ProtoBuf.ProtoMember(2, Name = @"channels")]
        public uint Channels
        {
            get { return __pbn__Channels.GetValueOrDefault(); }
            set { __pbn__Channels = value; }
        }
        public bool ShouldSerializeChannels() => __pbn__Channels != null;
        public void ResetChannels() => __pbn__Channels = null;
        private uint? __pbn__Channels;

        [global::ProtoBuf.ProtoMember(3, Name = @"height")]
        public uint Height
        {
            get { return __pbn__Height.GetValueOrDefault(); }
            set { __pbn__Height = value; }
        }
        public bool ShouldSerializeHeight() => __pbn__Height != null;
        public void ResetHeight() => __pbn__Height = null;
        private uint? __pbn__Height;

        [global::ProtoBuf.ProtoMember(4, Name = @"width")]
        public uint Width
        {
            get { return __pbn__Width.GetValueOrDefault(); }
            set { __pbn__Width = value; }
        }
        public bool ShouldSerializeWidth() => __pbn__Width != null;
        public void ResetWidth() => __pbn__Width = null;
        private uint? __pbn__Width;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class MVNParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"normalize_variance")]
        [global::System.ComponentModel.DefaultValue(true)]
        public bool NormalizeVariance
        {
            get { return __pbn__NormalizeVariance ?? true; }
            set { __pbn__NormalizeVariance = value; }
        }
        public bool ShouldSerializeNormalizeVariance() => __pbn__NormalizeVariance != null;
        public void ResetNormalizeVariance() => __pbn__NormalizeVariance = null;
        private bool? __pbn__NormalizeVariance;

        [global::ProtoBuf.ProtoMember(2, Name = @"across_channels")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool AcrossChannels
        {
            get { return __pbn__AcrossChannels ?? false; }
            set { __pbn__AcrossChannels = value; }
        }
        public bool ShouldSerializeAcrossChannels() => __pbn__AcrossChannels != null;
        public void ResetAcrossChannels() => __pbn__AcrossChannels = null;
        private bool? __pbn__AcrossChannels;

        [global::ProtoBuf.ProtoMember(3, Name = @"eps")]
        [global::System.ComponentModel.DefaultValue(1e-009f)]
        public float Eps
        {
            get { return __pbn__Eps ?? 1e-009f; }
            set { __pbn__Eps = value; }
        }
        public bool ShouldSerializeEps() => __pbn__Eps != null;
        public void ResetEps() => __pbn__Eps = null;
        private float? __pbn__Eps;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ParameterParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"shape")]
        public BlobShape Shape { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class PoolingParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"pool")]
        [global::System.ComponentModel.DefaultValue(PoolMethod.Max)]
        public PoolMethod Pool
        {
            get { return __pbn__Pool ?? PoolMethod.Max; }
            set { __pbn__Pool = value; }
        }
        public bool ShouldSerializePool() => __pbn__Pool != null;
        public void ResetPool() => __pbn__Pool = null;
        private PoolMethod? __pbn__Pool;

        [global::ProtoBuf.ProtoMember(4, Name = @"pad")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint Pad
        {
            get { return __pbn__Pad ?? 0; }
            set { __pbn__Pad = value; }
        }
        public bool ShouldSerializePad() => __pbn__Pad != null;
        public void ResetPad() => __pbn__Pad = null;
        private uint? __pbn__Pad;

        [global::ProtoBuf.ProtoMember(9, Name = @"pad_h")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint PadH
        {
            get { return __pbn__PadH ?? 0; }
            set { __pbn__PadH = value; }
        }
        public bool ShouldSerializePadH() => __pbn__PadH != null;
        public void ResetPadH() => __pbn__PadH = null;
        private uint? __pbn__PadH;

        [global::ProtoBuf.ProtoMember(10, Name = @"pad_w")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint PadW
        {
            get { return __pbn__PadW ?? 0; }
            set { __pbn__PadW = value; }
        }
        public bool ShouldSerializePadW() => __pbn__PadW != null;
        public void ResetPadW() => __pbn__PadW = null;
        private uint? __pbn__PadW;

        [global::ProtoBuf.ProtoMember(2, Name = @"kernel_size")]
        public uint KernelSize
        {
            get { return __pbn__KernelSize.GetValueOrDefault(); }
            set { __pbn__KernelSize = value; }
        }
        public bool ShouldSerializeKernelSize() => __pbn__KernelSize != null;
        public void ResetKernelSize() => __pbn__KernelSize = null;
        private uint? __pbn__KernelSize;

        [global::ProtoBuf.ProtoMember(5, Name = @"kernel_h")]
        public uint KernelH
        {
            get { return __pbn__KernelH.GetValueOrDefault(); }
            set { __pbn__KernelH = value; }
        }
        public bool ShouldSerializeKernelH() => __pbn__KernelH != null;
        public void ResetKernelH() => __pbn__KernelH = null;
        private uint? __pbn__KernelH;

        [global::ProtoBuf.ProtoMember(6, Name = @"kernel_w")]
        public uint KernelW
        {
            get { return __pbn__KernelW.GetValueOrDefault(); }
            set { __pbn__KernelW = value; }
        }
        public bool ShouldSerializeKernelW() => __pbn__KernelW != null;
        public void ResetKernelW() => __pbn__KernelW = null;
        private uint? __pbn__KernelW;

        [global::ProtoBuf.ProtoMember(3, Name = @"stride")]
        [global::System.ComponentModel.DefaultValue(1)]
        public uint Stride
        {
            get { return __pbn__Stride ?? 1; }
            set { __pbn__Stride = value; }
        }
        public bool ShouldSerializeStride() => __pbn__Stride != null;
        public void ResetStride() => __pbn__Stride = null;
        private uint? __pbn__Stride;

        [global::ProtoBuf.ProtoMember(7, Name = @"stride_h")]
        public uint StrideH
        {
            get { return __pbn__StrideH.GetValueOrDefault(); }
            set { __pbn__StrideH = value; }
        }
        public bool ShouldSerializeStrideH() => __pbn__StrideH != null;
        public void ResetStrideH() => __pbn__StrideH = null;
        private uint? __pbn__StrideH;

        [global::ProtoBuf.ProtoMember(8, Name = @"stride_w")]
        public uint StrideW
        {
            get { return __pbn__StrideW.GetValueOrDefault(); }
            set { __pbn__StrideW = value; }
        }
        public bool ShouldSerializeStrideW() => __pbn__StrideW != null;
        public void ResetStrideW() => __pbn__StrideW = null;
        private uint? __pbn__StrideW;

        [global::ProtoBuf.ProtoMember(11)]
        [global::System.ComponentModel.DefaultValue(Engine.Default)]
        public Engine engine
        {
            get { return __pbn__engine ?? Engine.Default; }
            set { __pbn__engine = value; }
        }
        public bool ShouldSerializeengine() => __pbn__engine != null;
        public void Resetengine() => __pbn__engine = null;
        private Engine? __pbn__engine;

        [global::ProtoBuf.ProtoMember(12, Name = @"global_pooling")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool GlobalPooling
        {
            get { return __pbn__GlobalPooling ?? false; }
            set { __pbn__GlobalPooling = value; }
        }
        public bool ShouldSerializeGlobalPooling() => __pbn__GlobalPooling != null;
        public void ResetGlobalPooling() => __pbn__GlobalPooling = null;
        private bool? __pbn__GlobalPooling;

        [global::ProtoBuf.ProtoContract()]
        public enum PoolMethod
        {
            [global::ProtoBuf.ProtoEnum(Name = @"MAX")]
            Max = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"AVE")]
            Ave = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"STOCHASTIC")]
            Stochastic = 2,
        }

        [global::ProtoBuf.ProtoContract()]
        public enum Engine
        {
            [global::ProtoBuf.ProtoEnum(Name = @"DEFAULT")]
            Default = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"CAFFE")]
            Caffe = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"CUDNN")]
            Cudnn = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class PowerParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"power")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Power
        {
            get { return __pbn__Power ?? 1; }
            set { __pbn__Power = value; }
        }
        public bool ShouldSerializePower() => __pbn__Power != null;
        public void ResetPower() => __pbn__Power = null;
        private float? __pbn__Power;

        [global::ProtoBuf.ProtoMember(2, Name = @"scale")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Scale
        {
            get { return __pbn__Scale ?? 1; }
            set { __pbn__Scale = value; }
        }
        public bool ShouldSerializeScale() => __pbn__Scale != null;
        public void ResetScale() => __pbn__Scale = null;
        private float? __pbn__Scale;

        [global::ProtoBuf.ProtoMember(3, Name = @"shift")]
        [global::System.ComponentModel.DefaultValue(0)]
        public float Shift
        {
            get { return __pbn__Shift ?? 0; }
            set { __pbn__Shift = value; }
        }
        public bool ShouldSerializeShift() => __pbn__Shift != null;
        public void ResetShift() => __pbn__Shift = null;
        private float? __pbn__Shift;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class PythonParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"module")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Module
        {
            get { return __pbn__Module ?? ""; }
            set { __pbn__Module = value; }
        }
        public bool ShouldSerializeModule() => __pbn__Module != null;
        public void ResetModule() => __pbn__Module = null;
        private string __pbn__Module;

        [global::ProtoBuf.ProtoMember(2, Name = @"layer")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Layer
        {
            get { return __pbn__Layer ?? ""; }
            set { __pbn__Layer = value; }
        }
        public bool ShouldSerializeLayer() => __pbn__Layer != null;
        public void ResetLayer() => __pbn__Layer = null;
        private string __pbn__Layer;

        [global::ProtoBuf.ProtoMember(3, Name = @"param_str")]
        [global::System.ComponentModel.DefaultValue("")]
        public string ParamStr
        {
            get { return __pbn__ParamStr ?? ""; }
            set { __pbn__ParamStr = value; }
        }
        public bool ShouldSerializeParamStr() => __pbn__ParamStr != null;
        public void ResetParamStr() => __pbn__ParamStr = null;
        private string __pbn__ParamStr;

        [global::ProtoBuf.ProtoMember(4, Name = @"share_in_parallel")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool ShareInParallel
        {
            get { return __pbn__ShareInParallel ?? false; }
            set { __pbn__ShareInParallel = value; }
        }
        public bool ShouldSerializeShareInParallel() => __pbn__ShareInParallel != null;
        public void ResetShareInParallel() => __pbn__ShareInParallel = null;
        private bool? __pbn__ShareInParallel;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class RecurrentParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"num_output")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint NumOutput
        {
            get { return __pbn__NumOutput ?? 0; }
            set { __pbn__NumOutput = value; }
        }
        public bool ShouldSerializeNumOutput() => __pbn__NumOutput != null;
        public void ResetNumOutput() => __pbn__NumOutput = null;
        private uint? __pbn__NumOutput;

        [global::ProtoBuf.ProtoMember(2, Name = @"weight_filler")]
        public FillerParameter WeightFiller { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"bias_filler")]
        public FillerParameter BiasFiller { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"debug_info")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool DebugInfo
        {
            get { return __pbn__DebugInfo ?? false; }
            set { __pbn__DebugInfo = value; }
        }
        public bool ShouldSerializeDebugInfo() => __pbn__DebugInfo != null;
        public void ResetDebugInfo() => __pbn__DebugInfo = null;
        private bool? __pbn__DebugInfo;

        [global::ProtoBuf.ProtoMember(5, Name = @"expose_hidden")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool ExposeHidden
        {
            get { return __pbn__ExposeHidden ?? false; }
            set { __pbn__ExposeHidden = value; }
        }
        public bool ShouldSerializeExposeHidden() => __pbn__ExposeHidden != null;
        public void ResetExposeHidden() => __pbn__ExposeHidden = null;
        private bool? __pbn__ExposeHidden;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ReductionParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"operation")]
        [global::System.ComponentModel.DefaultValue(ReductionOp.Sum)]
        public ReductionOp Operation
        {
            get { return __pbn__Operation ?? ReductionOp.Sum; }
            set { __pbn__Operation = value; }
        }
        public bool ShouldSerializeOperation() => __pbn__Operation != null;
        public void ResetOperation() => __pbn__Operation = null;
        private ReductionOp? __pbn__Operation;

        [global::ProtoBuf.ProtoMember(2, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int Axis
        {
            get { return __pbn__Axis ?? 0; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(3, Name = @"coeff")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Coeff
        {
            get { return __pbn__Coeff ?? 1; }
            set { __pbn__Coeff = value; }
        }
        public bool ShouldSerializeCoeff() => __pbn__Coeff != null;
        public void ResetCoeff() => __pbn__Coeff = null;
        private float? __pbn__Coeff;

        [global::ProtoBuf.ProtoContract()]
        public enum ReductionOp
        {
            [global::ProtoBuf.ProtoEnum(Name = @"SUM")]
            Sum = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"ASUM")]
            Asum = 2,
            [global::ProtoBuf.ProtoEnum(Name = @"SUMSQ")]
            Sumsq = 3,
            [global::ProtoBuf.ProtoEnum(Name = @"MEAN")]
            Mean = 4,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ReLUParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"negative_slope")]
        [global::System.ComponentModel.DefaultValue(0)]
        public float NegativeSlope
        {
            get { return __pbn__NegativeSlope ?? 0; }
            set { __pbn__NegativeSlope = value; }
        }
        public bool ShouldSerializeNegativeSlope() => __pbn__NegativeSlope != null;
        public void ResetNegativeSlope() => __pbn__NegativeSlope = null;
        private float? __pbn__NegativeSlope;

        [global::ProtoBuf.ProtoMember(2)]
        [global::System.ComponentModel.DefaultValue(Engine.Default)]
        public Engine engine
        {
            get { return __pbn__engine ?? Engine.Default; }
            set { __pbn__engine = value; }
        }
        public bool ShouldSerializeengine() => __pbn__engine != null;
        public void Resetengine() => __pbn__engine = null;
        private Engine? __pbn__engine;

        [global::ProtoBuf.ProtoContract()]
        public enum Engine
        {
            [global::ProtoBuf.ProtoEnum(Name = @"DEFAULT")]
            Default = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"CAFFE")]
            Caffe = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"CUDNN")]
            Cudnn = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ReshapeParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"shape")]
        public BlobShape Shape { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int Axis
        {
            get { return __pbn__Axis ?? 0; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(3, Name = @"num_axes")]
        [global::System.ComponentModel.DefaultValue(-1)]
        public int NumAxes
        {
            get { return __pbn__NumAxes ?? -1; }
            set { __pbn__NumAxes = value; }
        }
        public bool ShouldSerializeNumAxes() => __pbn__NumAxes != null;
        public void ResetNumAxes() => __pbn__NumAxes = null;
        private int? __pbn__NumAxes;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ScaleParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(2, Name = @"num_axes")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int NumAxes
        {
            get { return __pbn__NumAxes ?? 1; }
            set { __pbn__NumAxes = value; }
        }
        public bool ShouldSerializeNumAxes() => __pbn__NumAxes != null;
        public void ResetNumAxes() => __pbn__NumAxes = null;
        private int? __pbn__NumAxes;

        [global::ProtoBuf.ProtoMember(3, Name = @"filler")]
        public FillerParameter Filler { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"bias_term")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool BiasTerm
        {
            get { return __pbn__BiasTerm ?? false; }
            set { __pbn__BiasTerm = value; }
        }
        public bool ShouldSerializeBiasTerm() => __pbn__BiasTerm != null;
        public void ResetBiasTerm() => __pbn__BiasTerm = null;
        private bool? __pbn__BiasTerm;

        [global::ProtoBuf.ProtoMember(5, Name = @"bias_filler")]
        public FillerParameter BiasFiller { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class SigmoidParameter
    {
        [global::ProtoBuf.ProtoMember(1)]
        [global::System.ComponentModel.DefaultValue(Engine.Default)]
        public Engine engine
        {
            get { return __pbn__engine ?? Engine.Default; }
            set { __pbn__engine = value; }
        }
        public bool ShouldSerializeengine() => __pbn__engine != null;
        public void Resetengine() => __pbn__engine = null;
        private Engine? __pbn__engine;

        [global::ProtoBuf.ProtoContract()]
        public enum Engine
        {
            [global::ProtoBuf.ProtoEnum(Name = @"DEFAULT")]
            Default = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"CAFFE")]
            Caffe = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"CUDNN")]
            Cudnn = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class SliceParameter
    {
        [global::ProtoBuf.ProtoMember(3, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(2, Name = @"slice_point")]
        public uint[] SlicePoints { get; set; }

        [global::ProtoBuf.ProtoMember(1, Name = @"slice_dim")]
        [global::System.ComponentModel.DefaultValue(1)]
        public uint SliceDim
        {
            get { return __pbn__SliceDim ?? 1; }
            set { __pbn__SliceDim = value; }
        }
        public bool ShouldSerializeSliceDim() => __pbn__SliceDim != null;
        public void ResetSliceDim() => __pbn__SliceDim = null;
        private uint? __pbn__SliceDim;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class SoftmaxParameter
    {
        [global::ProtoBuf.ProtoMember(1)]
        [global::System.ComponentModel.DefaultValue(Engine.Default)]
        public Engine engine
        {
            get { return __pbn__engine ?? Engine.Default; }
            set { __pbn__engine = value; }
        }
        public bool ShouldSerializeengine() => __pbn__engine != null;
        public void Resetengine() => __pbn__engine = null;
        private Engine? __pbn__engine;

        [global::ProtoBuf.ProtoMember(2, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoContract()]
        public enum Engine
        {
            [global::ProtoBuf.ProtoEnum(Name = @"DEFAULT")]
            Default = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"CAFFE")]
            Caffe = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"CUDNN")]
            Cudnn = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class TanHParameter
    {
        [global::ProtoBuf.ProtoMember(1)]
        [global::System.ComponentModel.DefaultValue(Engine.Default)]
        public Engine engine
        {
            get { return __pbn__engine ?? Engine.Default; }
            set { __pbn__engine = value; }
        }
        public bool ShouldSerializeengine() => __pbn__engine != null;
        public void Resetengine() => __pbn__engine = null;
        private Engine? __pbn__engine;

        [global::ProtoBuf.ProtoContract()]
        public enum Engine
        {
            [global::ProtoBuf.ProtoEnum(Name = @"DEFAULT")]
            Default = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"CAFFE")]
            Caffe = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"CUDNN")]
            Cudnn = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class TileParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"axis")]
        [global::System.ComponentModel.DefaultValue(1)]
        public int Axis
        {
            get { return __pbn__Axis ?? 1; }
            set { __pbn__Axis = value; }
        }
        public bool ShouldSerializeAxis() => __pbn__Axis != null;
        public void ResetAxis() => __pbn__Axis = null;
        private int? __pbn__Axis;

        [global::ProtoBuf.ProtoMember(2, Name = @"tiles")]
        public int Tiles
        {
            get { return __pbn__Tiles.GetValueOrDefault(); }
            set { __pbn__Tiles = value; }
        }
        public bool ShouldSerializeTiles() => __pbn__Tiles != null;
        public void ResetTiles() => __pbn__Tiles = null;
        private int? __pbn__Tiles;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ThresholdParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"threshold")]
        [global::System.ComponentModel.DefaultValue(0)]
        public float Threshold
        {
            get { return __pbn__Threshold ?? 0; }
            set { __pbn__Threshold = value; }
        }
        public bool ShouldSerializeThreshold() => __pbn__Threshold != null;
        public void ResetThreshold() => __pbn__Threshold = null;
        private float? __pbn__Threshold;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class WindowDataParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"source")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Source
        {
            get { return __pbn__Source ?? ""; }
            set { __pbn__Source = value; }
        }
        public bool ShouldSerializeSource() => __pbn__Source != null;
        public void ResetSource() => __pbn__Source = null;
        private string __pbn__Source;

        [global::ProtoBuf.ProtoMember(2, Name = @"scale")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Scale
        {
            get { return __pbn__Scale ?? 1; }
            set { __pbn__Scale = value; }
        }
        public bool ShouldSerializeScale() => __pbn__Scale != null;
        public void ResetScale() => __pbn__Scale = null;
        private float? __pbn__Scale;

        [global::ProtoBuf.ProtoMember(3, Name = @"mean_file")]
        [global::System.ComponentModel.DefaultValue("")]
        public string MeanFile
        {
            get { return __pbn__MeanFile ?? ""; }
            set { __pbn__MeanFile = value; }
        }
        public bool ShouldSerializeMeanFile() => __pbn__MeanFile != null;
        public void ResetMeanFile() => __pbn__MeanFile = null;
        private string __pbn__MeanFile;

        [global::ProtoBuf.ProtoMember(4, Name = @"batch_size")]
        public uint BatchSize
        {
            get { return __pbn__BatchSize.GetValueOrDefault(); }
            set { __pbn__BatchSize = value; }
        }
        public bool ShouldSerializeBatchSize() => __pbn__BatchSize != null;
        public void ResetBatchSize() => __pbn__BatchSize = null;
        private uint? __pbn__BatchSize;

        [global::ProtoBuf.ProtoMember(5, Name = @"crop_size")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint CropSize
        {
            get { return __pbn__CropSize ?? 0; }
            set { __pbn__CropSize = value; }
        }
        public bool ShouldSerializeCropSize() => __pbn__CropSize != null;
        public void ResetCropSize() => __pbn__CropSize = null;
        private uint? __pbn__CropSize;

        [global::ProtoBuf.ProtoMember(6, Name = @"mirror")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool Mirror
        {
            get { return __pbn__Mirror ?? false; }
            set { __pbn__Mirror = value; }
        }
        public bool ShouldSerializeMirror() => __pbn__Mirror != null;
        public void ResetMirror() => __pbn__Mirror = null;
        private bool? __pbn__Mirror;

        [global::ProtoBuf.ProtoMember(7, Name = @"fg_threshold")]
        [global::System.ComponentModel.DefaultValue(0.5f)]
        public float FgThreshold
        {
            get { return __pbn__FgThreshold ?? 0.5f; }
            set { __pbn__FgThreshold = value; }
        }
        public bool ShouldSerializeFgThreshold() => __pbn__FgThreshold != null;
        public void ResetFgThreshold() => __pbn__FgThreshold = null;
        private float? __pbn__FgThreshold;

        [global::ProtoBuf.ProtoMember(8, Name = @"bg_threshold")]
        [global::System.ComponentModel.DefaultValue(0.5f)]
        public float BgThreshold
        {
            get { return __pbn__BgThreshold ?? 0.5f; }
            set { __pbn__BgThreshold = value; }
        }
        public bool ShouldSerializeBgThreshold() => __pbn__BgThreshold != null;
        public void ResetBgThreshold() => __pbn__BgThreshold = null;
        private float? __pbn__BgThreshold;

        [global::ProtoBuf.ProtoMember(9, Name = @"fg_fraction")]
        [global::System.ComponentModel.DefaultValue(0.25f)]
        public float FgFraction
        {
            get { return __pbn__FgFraction ?? 0.25f; }
            set { __pbn__FgFraction = value; }
        }
        public bool ShouldSerializeFgFraction() => __pbn__FgFraction != null;
        public void ResetFgFraction() => __pbn__FgFraction = null;
        private float? __pbn__FgFraction;

        [global::ProtoBuf.ProtoMember(10, Name = @"context_pad")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint ContextPad
        {
            get { return __pbn__ContextPad ?? 0; }
            set { __pbn__ContextPad = value; }
        }
        public bool ShouldSerializeContextPad() => __pbn__ContextPad != null;
        public void ResetContextPad() => __pbn__ContextPad = null;
        private uint? __pbn__ContextPad;

        [global::ProtoBuf.ProtoMember(11, Name = @"crop_mode")]
        [global::System.ComponentModel.DefaultValue(@"warp")]
        public string CropMode
        {
            get { return __pbn__CropMode ?? @"warp"; }
            set { __pbn__CropMode = value; }
        }
        public bool ShouldSerializeCropMode() => __pbn__CropMode != null;
        public void ResetCropMode() => __pbn__CropMode = null;
        private string __pbn__CropMode;

        [global::ProtoBuf.ProtoMember(12, Name = @"cache_images")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool CacheImages
        {
            get { return __pbn__CacheImages ?? false; }
            set { __pbn__CacheImages = value; }
        }
        public bool ShouldSerializeCacheImages() => __pbn__CacheImages != null;
        public void ResetCacheImages() => __pbn__CacheImages = null;
        private bool? __pbn__CacheImages;

        [global::ProtoBuf.ProtoMember(13, Name = @"root_folder")]
        [global::System.ComponentModel.DefaultValue("")]
        public string RootFolder
        {
            get { return __pbn__RootFolder ?? ""; }
            set { __pbn__RootFolder = value; }
        }
        public bool ShouldSerializeRootFolder() => __pbn__RootFolder != null;
        public void ResetRootFolder() => __pbn__RootFolder = null;
        private string __pbn__RootFolder;

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class SPPParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"pyramid_height")]
        public uint PyramidHeight
        {
            get { return __pbn__PyramidHeight.GetValueOrDefault(); }
            set { __pbn__PyramidHeight = value; }
        }
        public bool ShouldSerializePyramidHeight() => __pbn__PyramidHeight != null;
        public void ResetPyramidHeight() => __pbn__PyramidHeight = null;
        private uint? __pbn__PyramidHeight;

        [global::ProtoBuf.ProtoMember(2, Name = @"pool")]
        [global::System.ComponentModel.DefaultValue(PoolMethod.Max)]
        public PoolMethod Pool
        {
            get { return __pbn__Pool ?? PoolMethod.Max; }
            set { __pbn__Pool = value; }
        }
        public bool ShouldSerializePool() => __pbn__Pool != null;
        public void ResetPool() => __pbn__Pool = null;
        private PoolMethod? __pbn__Pool;

        [global::ProtoBuf.ProtoMember(6)]
        [global::System.ComponentModel.DefaultValue(Engine.Default)]
        public Engine engine
        {
            get { return __pbn__engine ?? Engine.Default; }
            set { __pbn__engine = value; }
        }
        public bool ShouldSerializeengine() => __pbn__engine != null;
        public void Resetengine() => __pbn__engine = null;
        private Engine? __pbn__engine;

        [global::ProtoBuf.ProtoContract()]
        public enum PoolMethod
        {
            [global::ProtoBuf.ProtoEnum(Name = @"MAX")]
            Max = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"AVE")]
            Ave = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"STOCHASTIC")]
            Stochastic = 2,
        }

        [global::ProtoBuf.ProtoContract()]
        public enum Engine
        {
            [global::ProtoBuf.ProtoEnum(Name = @"DEFAULT")]
            Default = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"CAFFE")]
            Caffe = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"CUDNN")]
            Cudnn = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class V1LayerParameter
    {
        [global::ProtoBuf.ProtoMember(2, Name = @"bottom")]
        public global::System.Collections.Generic.List<string> Bottoms { get; } = new global::System.Collections.Generic.List<string>();

        [global::ProtoBuf.ProtoMember(3, Name = @"top")]
        public global::System.Collections.Generic.List<string> Tops { get; } = new global::System.Collections.Generic.List<string>();

        [global::ProtoBuf.ProtoMember(4, Name = @"name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Name
        {
            get { return __pbn__Name ?? ""; }
            set { __pbn__Name = value; }
        }
        public bool ShouldSerializeName() => __pbn__Name != null;
        public void ResetName() => __pbn__Name = null;
        private string __pbn__Name;

        [global::ProtoBuf.ProtoMember(32, Name = @"include")]
        public global::System.Collections.Generic.List<NetStateRule> Includes { get; } = new global::System.Collections.Generic.List<NetStateRule>();

        [global::ProtoBuf.ProtoMember(33, Name = @"exclude")]
        public global::System.Collections.Generic.List<NetStateRule> Excludes { get; } = new global::System.Collections.Generic.List<NetStateRule>();

        [global::ProtoBuf.ProtoMember(5, Name = @"type")]
        [global::System.ComponentModel.DefaultValue(LayerType.None)]
        public LayerType Type
        {
            get { return __pbn__Type ?? LayerType.None; }
            set { __pbn__Type = value; }
        }
        public bool ShouldSerializeType() => __pbn__Type != null;
        public void ResetType() => __pbn__Type = null;
        private LayerType? __pbn__Type;

        [global::ProtoBuf.ProtoMember(6, Name = @"blobs")]
        public global::System.Collections.Generic.List<BlobProto> Blobs { get; } = new global::System.Collections.Generic.List<BlobProto>();

        [global::ProtoBuf.ProtoMember(1001, Name = @"param")]
        public global::System.Collections.Generic.List<string> Params { get; } = new global::System.Collections.Generic.List<string>();

        [global::ProtoBuf.ProtoMember(1002, Name = @"blob_share_mode")]
        public global::System.Collections.Generic.List<DimCheckMode> BlobShareModes { get; } = new global::System.Collections.Generic.List<DimCheckMode>();

        [global::ProtoBuf.ProtoMember(7, Name = @"blobs_lr")]
        public float[] BlobsLrs { get; set; }

        [global::ProtoBuf.ProtoMember(8, Name = @"weight_decay")]
        public float[] WeightDecays { get; set; }

        [global::ProtoBuf.ProtoMember(35, Name = @"loss_weight")]
        public float[] LossWeights { get; set; }

        [global::ProtoBuf.ProtoMember(27, Name = @"accuracy_param")]
        public AccuracyParameter AccuracyParam { get; set; }

        [global::ProtoBuf.ProtoMember(23, Name = @"argmax_param")]
        public ArgMaxParameter ArgmaxParam { get; set; }

        [global::ProtoBuf.ProtoMember(9, Name = @"concat_param")]
        public ConcatParameter ConcatParam { get; set; }

        [global::ProtoBuf.ProtoMember(40, Name = @"contrastive_loss_param")]
        public ContrastiveLossParameter ContrastiveLossParam { get; set; }

        [global::ProtoBuf.ProtoMember(10, Name = @"convolution_param")]
        public ConvolutionParameter ConvolutionParam { get; set; }

        [global::ProtoBuf.ProtoMember(11, Name = @"data_param")]
        public DataParameter DataParam { get; set; }

        [global::ProtoBuf.ProtoMember(12, Name = @"dropout_param")]
        public DropoutParameter DropoutParam { get; set; }

        [global::ProtoBuf.ProtoMember(26, Name = @"dummy_data_param")]
        public DummyDataParameter DummyDataParam { get; set; }

        [global::ProtoBuf.ProtoMember(24, Name = @"eltwise_param")]
        public EltwiseParameter EltwiseParam { get; set; }

        [global::ProtoBuf.ProtoMember(41, Name = @"exp_param")]
        public ExpParameter ExpParam { get; set; }

        [global::ProtoBuf.ProtoMember(13, Name = @"hdf5_data_param")]
        public HDF5DataParameter Hdf5DataParam { get; set; }

        [global::ProtoBuf.ProtoMember(14, Name = @"hdf5_output_param")]
        public HDF5OutputParameter Hdf5OutputParam { get; set; }

        [global::ProtoBuf.ProtoMember(29, Name = @"hinge_loss_param")]
        public HingeLossParameter HingeLossParam { get; set; }

        [global::ProtoBuf.ProtoMember(15, Name = @"image_data_param")]
        public ImageDataParameter ImageDataParam { get; set; }

        [global::ProtoBuf.ProtoMember(16, Name = @"infogain_loss_param")]
        public InfogainLossParameter InfogainLossParam { get; set; }

        [global::ProtoBuf.ProtoMember(17, Name = @"inner_product_param")]
        public InnerProductParameter InnerProductParam { get; set; }

        [global::ProtoBuf.ProtoMember(18, Name = @"lrn_param")]
        public LRNParameter LrnParam { get; set; }

        [global::ProtoBuf.ProtoMember(22, Name = @"memory_data_param")]
        public MemoryDataParameter MemoryDataParam { get; set; }

        [global::ProtoBuf.ProtoMember(34, Name = @"mvn_param")]
        public MVNParameter MvnParam { get; set; }

        [global::ProtoBuf.ProtoMember(19, Name = @"pooling_param")]
        public PoolingParameter PoolingParam { get; set; }

        [global::ProtoBuf.ProtoMember(21, Name = @"power_param")]
        public PowerParameter PowerParam { get; set; }

        [global::ProtoBuf.ProtoMember(30, Name = @"relu_param")]
        public ReLUParameter ReluParam { get; set; }

        [global::ProtoBuf.ProtoMember(38, Name = @"sigmoid_param")]
        public SigmoidParameter SigmoidParam { get; set; }

        [global::ProtoBuf.ProtoMember(39, Name = @"softmax_param")]
        public SoftmaxParameter SoftmaxParam { get; set; }

        [global::ProtoBuf.ProtoMember(31, Name = @"slice_param")]
        public SliceParameter SliceParam { get; set; }

        [global::ProtoBuf.ProtoMember(37, Name = @"tanh_param")]
        public TanHParameter TanhParam { get; set; }

        [global::ProtoBuf.ProtoMember(25, Name = @"threshold_param")]
        public ThresholdParameter ThresholdParam { get; set; }

        [global::ProtoBuf.ProtoMember(20, Name = @"window_data_param")]
        public WindowDataParameter WindowDataParam { get; set; }

        [global::ProtoBuf.ProtoMember(36, Name = @"transform_param")]
        public TransformationParameter TransformParam { get; set; }

        [global::ProtoBuf.ProtoMember(42, Name = @"loss_param")]
        public LossParameter LossParam { get; set; }

        [global::ProtoBuf.ProtoMember(1, Name = @"layer")]
        public V0LayerParameter Layer { get; set; }

        [global::ProtoBuf.ProtoContract()]
        public enum LayerType
        {
            [global::ProtoBuf.ProtoEnum(Name = @"NONE")]
            None = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"ABSVAL")]
            Absval = 35,
            [global::ProtoBuf.ProtoEnum(Name = @"ACCURACY")]
            Accuracy = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"ARGMAX")]
            Argmax = 30,
            [global::ProtoBuf.ProtoEnum(Name = @"BNLL")]
            Bnll = 2,
            [global::ProtoBuf.ProtoEnum(Name = @"CONCAT")]
            Concat = 3,
            [global::ProtoBuf.ProtoEnum(Name = @"CONTRASTIVE_LOSS")]
            ContrastiveLoss = 37,
            [global::ProtoBuf.ProtoEnum(Name = @"CONVOLUTION")]
            Convolution = 4,
            [global::ProtoBuf.ProtoEnum(Name = @"DATA")]
            Data = 5,
            [global::ProtoBuf.ProtoEnum(Name = @"DECONVOLUTION")]
            Deconvolution = 39,
            [global::ProtoBuf.ProtoEnum(Name = @"DROPOUT")]
            Dropout = 6,
            [global::ProtoBuf.ProtoEnum(Name = @"DUMMY_DATA")]
            DummyData = 32,
            [global::ProtoBuf.ProtoEnum(Name = @"EUCLIDEAN_LOSS")]
            EuclideanLoss = 7,
            [global::ProtoBuf.ProtoEnum(Name = @"ELTWISE")]
            Eltwise = 25,
            [global::ProtoBuf.ProtoEnum(Name = @"EXP")]
            Exp = 38,
            [global::ProtoBuf.ProtoEnum(Name = @"FLATTEN")]
            Flatten = 8,
            [global::ProtoBuf.ProtoEnum(Name = @"HDF5_DATA")]
            Hdf5Data = 9,
            [global::ProtoBuf.ProtoEnum(Name = @"HDF5_OUTPUT")]
            Hdf5Output = 10,
            [global::ProtoBuf.ProtoEnum(Name = @"HINGE_LOSS")]
            HingeLoss = 28,
            [global::ProtoBuf.ProtoEnum(Name = @"IM2COL")]
            Im2col = 11,
            [global::ProtoBuf.ProtoEnum(Name = @"IMAGE_DATA")]
            ImageData = 12,
            [global::ProtoBuf.ProtoEnum(Name = @"INFOGAIN_LOSS")]
            InfogainLoss = 13,
            [global::ProtoBuf.ProtoEnum(Name = @"INNER_PRODUCT")]
            InnerProduct = 14,
            [global::ProtoBuf.ProtoEnum(Name = @"LRN")]
            Lrn = 15,
            [global::ProtoBuf.ProtoEnum(Name = @"MEMORY_DATA")]
            MemoryData = 29,
            [global::ProtoBuf.ProtoEnum(Name = @"MULTINOMIAL_LOGISTIC_LOSS")]
            MultinomialLogisticLoss = 16,
            [global::ProtoBuf.ProtoEnum(Name = @"MVN")]
            Mvn = 34,
            [global::ProtoBuf.ProtoEnum(Name = @"POOLING")]
            Pooling = 17,
            [global::ProtoBuf.ProtoEnum(Name = @"POWER")]
            Power = 26,
            [global::ProtoBuf.ProtoEnum(Name = @"RELU")]
            Relu = 18,
            [global::ProtoBuf.ProtoEnum(Name = @"SIGMOID")]
            Sigmoid = 19,
            [global::ProtoBuf.ProtoEnum(Name = @"SIGMOID_CROSS_ENTROPY_LOSS")]
            SigmoidCrossEntropyLoss = 27,
            [global::ProtoBuf.ProtoEnum(Name = @"SILENCE")]
            Silence = 36,
            [global::ProtoBuf.ProtoEnum(Name = @"SOFTMAX")]
            Softmax = 20,
            [global::ProtoBuf.ProtoEnum(Name = @"SOFTMAX_LOSS")]
            SoftmaxLoss = 21,
            [global::ProtoBuf.ProtoEnum(Name = @"SPLIT")]
            Split = 22,
            [global::ProtoBuf.ProtoEnum(Name = @"SLICE")]
            Slice = 33,
            [global::ProtoBuf.ProtoEnum(Name = @"TANH")]
            Tanh = 23,
            [global::ProtoBuf.ProtoEnum(Name = @"WINDOW_DATA")]
            WindowData = 24,
            [global::ProtoBuf.ProtoEnum(Name = @"THRESHOLD")]
            Threshold = 31,
        }

        [global::ProtoBuf.ProtoContract()]
        public enum DimCheckMode
        {
            [global::ProtoBuf.ProtoEnum(Name = @"STRICT")]
            Strict = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"PERMISSIVE")]
            Permissive = 1,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class V0LayerParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Name
        {
            get { return __pbn__Name ?? ""; }
            set { __pbn__Name = value; }
        }
        public bool ShouldSerializeName() => __pbn__Name != null;
        public void ResetName() => __pbn__Name = null;
        private string __pbn__Name;

        [global::ProtoBuf.ProtoMember(2, Name = @"type")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Type
        {
            get { return __pbn__Type ?? ""; }
            set { __pbn__Type = value; }
        }
        public bool ShouldSerializeType() => __pbn__Type != null;
        public void ResetType() => __pbn__Type = null;
        private string __pbn__Type;

        [global::ProtoBuf.ProtoMember(3, Name = @"num_output")]
        public uint NumOutput
        {
            get { return __pbn__NumOutput.GetValueOrDefault(); }
            set { __pbn__NumOutput = value; }
        }
        public bool ShouldSerializeNumOutput() => __pbn__NumOutput != null;
        public void ResetNumOutput() => __pbn__NumOutput = null;
        private uint? __pbn__NumOutput;

        [global::ProtoBuf.ProtoMember(4, Name = @"biasterm")]
        [global::System.ComponentModel.DefaultValue(true)]
        public bool Biasterm
        {
            get { return __pbn__Biasterm ?? true; }
            set { __pbn__Biasterm = value; }
        }
        public bool ShouldSerializeBiasterm() => __pbn__Biasterm != null;
        public void ResetBiasterm() => __pbn__Biasterm = null;
        private bool? __pbn__Biasterm;

        [global::ProtoBuf.ProtoMember(5, Name = @"weight_filler")]
        public FillerParameter WeightFiller { get; set; }

        [global::ProtoBuf.ProtoMember(6, Name = @"bias_filler")]
        public FillerParameter BiasFiller { get; set; }

        [global::ProtoBuf.ProtoMember(7, Name = @"pad")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint Pad
        {
            get { return __pbn__Pad ?? 0; }
            set { __pbn__Pad = value; }
        }
        public bool ShouldSerializePad() => __pbn__Pad != null;
        public void ResetPad() => __pbn__Pad = null;
        private uint? __pbn__Pad;

        [global::ProtoBuf.ProtoMember(8, Name = @"kernelsize")]
        public uint Kernelsize
        {
            get { return __pbn__Kernelsize.GetValueOrDefault(); }
            set { __pbn__Kernelsize = value; }
        }
        public bool ShouldSerializeKernelsize() => __pbn__Kernelsize != null;
        public void ResetKernelsize() => __pbn__Kernelsize = null;
        private uint? __pbn__Kernelsize;

        [global::ProtoBuf.ProtoMember(9, Name = @"group")]
        [global::System.ComponentModel.DefaultValue(1)]
        public uint Group
        {
            get { return __pbn__Group ?? 1; }
            set { __pbn__Group = value; }
        }
        public bool ShouldSerializeGroup() => __pbn__Group != null;
        public void ResetGroup() => __pbn__Group = null;
        private uint? __pbn__Group;

        [global::ProtoBuf.ProtoMember(10, Name = @"stride")]
        [global::System.ComponentModel.DefaultValue(1)]
        public uint Stride
        {
            get { return __pbn__Stride ?? 1; }
            set { __pbn__Stride = value; }
        }
        public bool ShouldSerializeStride() => __pbn__Stride != null;
        public void ResetStride() => __pbn__Stride = null;
        private uint? __pbn__Stride;

        [global::ProtoBuf.ProtoMember(11, Name = @"pool")]
        [global::System.ComponentModel.DefaultValue(PoolMethod.Max)]
        public PoolMethod Pool
        {
            get { return __pbn__Pool ?? PoolMethod.Max; }
            set { __pbn__Pool = value; }
        }
        public bool ShouldSerializePool() => __pbn__Pool != null;
        public void ResetPool() => __pbn__Pool = null;
        private PoolMethod? __pbn__Pool;

        [global::ProtoBuf.ProtoMember(12, Name = @"dropout_ratio")]
        [global::System.ComponentModel.DefaultValue(0.5f)]
        public float DropoutRatio
        {
            get { return __pbn__DropoutRatio ?? 0.5f; }
            set { __pbn__DropoutRatio = value; }
        }
        public bool ShouldSerializeDropoutRatio() => __pbn__DropoutRatio != null;
        public void ResetDropoutRatio() => __pbn__DropoutRatio = null;
        private float? __pbn__DropoutRatio;

        [global::ProtoBuf.ProtoMember(13, Name = @"local_size")]
        [global::System.ComponentModel.DefaultValue(5)]
        public uint LocalSize
        {
            get { return __pbn__LocalSize ?? 5; }
            set { __pbn__LocalSize = value; }
        }
        public bool ShouldSerializeLocalSize() => __pbn__LocalSize != null;
        public void ResetLocalSize() => __pbn__LocalSize = null;
        private uint? __pbn__LocalSize;

        [global::ProtoBuf.ProtoMember(14, Name = @"alpha")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Alpha
        {
            get { return __pbn__Alpha ?? 1; }
            set { __pbn__Alpha = value; }
        }
        public bool ShouldSerializeAlpha() => __pbn__Alpha != null;
        public void ResetAlpha() => __pbn__Alpha = null;
        private float? __pbn__Alpha;

        [global::ProtoBuf.ProtoMember(15, Name = @"beta")]
        [global::System.ComponentModel.DefaultValue(0.75f)]
        public float Beta
        {
            get { return __pbn__Beta ?? 0.75f; }
            set { __pbn__Beta = value; }
        }
        public bool ShouldSerializeBeta() => __pbn__Beta != null;
        public void ResetBeta() => __pbn__Beta = null;
        private float? __pbn__Beta;

        [global::ProtoBuf.ProtoMember(22, Name = @"k")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float K
        {
            get { return __pbn__K ?? 1; }
            set { __pbn__K = value; }
        }
        public bool ShouldSerializeK() => __pbn__K != null;
        public void ResetK() => __pbn__K = null;
        private float? __pbn__K;

        [global::ProtoBuf.ProtoMember(16, Name = @"source")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Source
        {
            get { return __pbn__Source ?? ""; }
            set { __pbn__Source = value; }
        }
        public bool ShouldSerializeSource() => __pbn__Source != null;
        public void ResetSource() => __pbn__Source = null;
        private string __pbn__Source;

        [global::ProtoBuf.ProtoMember(17, Name = @"scale")]
        [global::System.ComponentModel.DefaultValue(1)]
        public float Scale
        {
            get { return __pbn__Scale ?? 1; }
            set { __pbn__Scale = value; }
        }
        public bool ShouldSerializeScale() => __pbn__Scale != null;
        public void ResetScale() => __pbn__Scale = null;
        private float? __pbn__Scale;

        [global::ProtoBuf.ProtoMember(18, Name = @"meanfile")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Meanfile
        {
            get { return __pbn__Meanfile ?? ""; }
            set { __pbn__Meanfile = value; }
        }
        public bool ShouldSerializeMeanfile() => __pbn__Meanfile != null;
        public void ResetMeanfile() => __pbn__Meanfile = null;
        private string __pbn__Meanfile;

        [global::ProtoBuf.ProtoMember(19, Name = @"batchsize")]
        public uint Batchsize
        {
            get { return __pbn__Batchsize.GetValueOrDefault(); }
            set { __pbn__Batchsize = value; }
        }
        public bool ShouldSerializeBatchsize() => __pbn__Batchsize != null;
        public void ResetBatchsize() => __pbn__Batchsize = null;
        private uint? __pbn__Batchsize;

        [global::ProtoBuf.ProtoMember(20, Name = @"cropsize")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint Cropsize
        {
            get { return __pbn__Cropsize ?? 0; }
            set { __pbn__Cropsize = value; }
        }
        public bool ShouldSerializeCropsize() => __pbn__Cropsize != null;
        public void ResetCropsize() => __pbn__Cropsize = null;
        private uint? __pbn__Cropsize;

        [global::ProtoBuf.ProtoMember(21, Name = @"mirror")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool Mirror
        {
            get { return __pbn__Mirror ?? false; }
            set { __pbn__Mirror = value; }
        }
        public bool ShouldSerializeMirror() => __pbn__Mirror != null;
        public void ResetMirror() => __pbn__Mirror = null;
        private bool? __pbn__Mirror;

        [global::ProtoBuf.ProtoMember(50, Name = @"blobs")]
        public global::System.Collections.Generic.List<BlobProto> Blobs { get; } = new global::System.Collections.Generic.List<BlobProto>();

        [global::ProtoBuf.ProtoMember(51, Name = @"blobs_lr")]
        public float[] BlobsLrs { get; set; }

        [global::ProtoBuf.ProtoMember(52, Name = @"weight_decay")]
        public float[] WeightDecays { get; set; }

        [global::ProtoBuf.ProtoMember(53, Name = @"rand_skip")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint RandSkip
        {
            get { return __pbn__RandSkip ?? 0; }
            set { __pbn__RandSkip = value; }
        }
        public bool ShouldSerializeRandSkip() => __pbn__RandSkip != null;
        public void ResetRandSkip() => __pbn__RandSkip = null;
        private uint? __pbn__RandSkip;

        [global::ProtoBuf.ProtoMember(54, Name = @"det_fg_threshold")]
        [global::System.ComponentModel.DefaultValue(0.5f)]
        public float DetFgThreshold
        {
            get { return __pbn__DetFgThreshold ?? 0.5f; }
            set { __pbn__DetFgThreshold = value; }
        }
        public bool ShouldSerializeDetFgThreshold() => __pbn__DetFgThreshold != null;
        public void ResetDetFgThreshold() => __pbn__DetFgThreshold = null;
        private float? __pbn__DetFgThreshold;

        [global::ProtoBuf.ProtoMember(55, Name = @"det_bg_threshold")]
        [global::System.ComponentModel.DefaultValue(0.5f)]
        public float DetBgThreshold
        {
            get { return __pbn__DetBgThreshold ?? 0.5f; }
            set { __pbn__DetBgThreshold = value; }
        }
        public bool ShouldSerializeDetBgThreshold() => __pbn__DetBgThreshold != null;
        public void ResetDetBgThreshold() => __pbn__DetBgThreshold = null;
        private float? __pbn__DetBgThreshold;

        [global::ProtoBuf.ProtoMember(56, Name = @"det_fg_fraction")]
        [global::System.ComponentModel.DefaultValue(0.25f)]
        public float DetFgFraction
        {
            get { return __pbn__DetFgFraction ?? 0.25f; }
            set { __pbn__DetFgFraction = value; }
        }
        public bool ShouldSerializeDetFgFraction() => __pbn__DetFgFraction != null;
        public void ResetDetFgFraction() => __pbn__DetFgFraction = null;
        private float? __pbn__DetFgFraction;

        [global::ProtoBuf.ProtoMember(58, Name = @"det_context_pad")]
        [global::System.ComponentModel.DefaultValue(0)]
        public uint DetContextPad
        {
            get { return __pbn__DetContextPad ?? 0; }
            set { __pbn__DetContextPad = value; }
        }
        public bool ShouldSerializeDetContextPad() => __pbn__DetContextPad != null;
        public void ResetDetContextPad() => __pbn__DetContextPad = null;
        private uint? __pbn__DetContextPad;

        [global::ProtoBuf.ProtoMember(59, Name = @"det_crop_mode")]
        [global::System.ComponentModel.DefaultValue(@"warp")]
        public string DetCropMode
        {
            get { return __pbn__DetCropMode ?? @"warp"; }
            set { __pbn__DetCropMode = value; }
        }
        public bool ShouldSerializeDetCropMode() => __pbn__DetCropMode != null;
        public void ResetDetCropMode() => __pbn__DetCropMode = null;
        private string __pbn__DetCropMode;

        [global::ProtoBuf.ProtoMember(60, Name = @"new_num")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int NewNum
        {
            get { return __pbn__NewNum ?? 0; }
            set { __pbn__NewNum = value; }
        }
        public bool ShouldSerializeNewNum() => __pbn__NewNum != null;
        public void ResetNewNum() => __pbn__NewNum = null;
        private int? __pbn__NewNum;

        [global::ProtoBuf.ProtoMember(61, Name = @"new_channels")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int NewChannels
        {
            get { return __pbn__NewChannels ?? 0; }
            set { __pbn__NewChannels = value; }
        }
        public bool ShouldSerializeNewChannels() => __pbn__NewChannels != null;
        public void ResetNewChannels() => __pbn__NewChannels = null;
        private int? __pbn__NewChannels;

        [global::ProtoBuf.ProtoMember(62, Name = @"new_height")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int NewHeight
        {
            get { return __pbn__NewHeight ?? 0; }
            set { __pbn__NewHeight = value; }
        }
        public bool ShouldSerializeNewHeight() => __pbn__NewHeight != null;
        public void ResetNewHeight() => __pbn__NewHeight = null;
        private int? __pbn__NewHeight;

        [global::ProtoBuf.ProtoMember(63, Name = @"new_width")]
        [global::System.ComponentModel.DefaultValue(0)]
        public int NewWidth
        {
            get { return __pbn__NewWidth ?? 0; }
            set { __pbn__NewWidth = value; }
        }
        public bool ShouldSerializeNewWidth() => __pbn__NewWidth != null;
        public void ResetNewWidth() => __pbn__NewWidth = null;
        private int? __pbn__NewWidth;

        [global::ProtoBuf.ProtoMember(64, Name = @"shuffle_images")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool ShuffleImages
        {
            get { return __pbn__ShuffleImages ?? false; }
            set { __pbn__ShuffleImages = value; }
        }
        public bool ShouldSerializeShuffleImages() => __pbn__ShuffleImages != null;
        public void ResetShuffleImages() => __pbn__ShuffleImages = null;
        private bool? __pbn__ShuffleImages;

        [global::ProtoBuf.ProtoMember(65, Name = @"concat_dim")]
        [global::System.ComponentModel.DefaultValue(1)]
        public uint ConcatDim
        {
            get { return __pbn__ConcatDim ?? 1; }
            set { __pbn__ConcatDim = value; }
        }
        public bool ShouldSerializeConcatDim() => __pbn__ConcatDim != null;
        public void ResetConcatDim() => __pbn__ConcatDim = null;
        private uint? __pbn__ConcatDim;

        [global::ProtoBuf.ProtoMember(1001, Name = @"hdf5_output_param")]
        public HDF5OutputParameter Hdf5OutputParam { get; set; }

        [global::ProtoBuf.ProtoContract()]
        public enum PoolMethod
        {
            [global::ProtoBuf.ProtoEnum(Name = @"MAX")]
            Max = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"AVE")]
            Ave = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"STOCHASTIC")]
            Stochastic = 2,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class PReLUParameter
    {
        [global::ProtoBuf.ProtoMember(1, Name = @"filler")]
        public FillerParameter Filler { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"channel_shared")]
        [global::System.ComponentModel.DefaultValue(false)]
        public bool ChannelShared
        {
            get { return __pbn__ChannelShared ?? false; }
            set { __pbn__ChannelShared = value; }
        }
        public bool ShouldSerializeChannelShared() => __pbn__ChannelShared != null;
        public void ResetChannelShared() => __pbn__ChannelShared = null;
        private bool? __pbn__ChannelShared;

    }

    [global::ProtoBuf.ProtoContract()]
    public enum Phase
    {
        [global::ProtoBuf.ProtoEnum(Name = @"TRAIN")]
        Train = 0,
        [global::ProtoBuf.ProtoEnum(Name = @"TEST")]
        Test = 1,
    }

}

#pragma warning restore CS1591, CS0612, CS3021
