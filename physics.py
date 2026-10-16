"""Extracted from OpenFWI_Para/FWIForward.py (local source, no license supplied).

Only the acoustic simulation path is retained. No license is inferred or granted
by this extraction. Deepwave PML is not the released OpenFWI damping ABC.
The source's x,z transpose and 1001 -> 1000 alignment are preserved.
"""
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional
import deepwave
import torch

class AbstractSourceGenerator(ABC):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def generate_amplitudes(
        self, nt: int, dt: float, offset_time: float, source_locations: torch.Tensor
    ) -> torch.Tensor:
        pass


class OpenFWIRickerSourceGenerator(AbstractSourceGenerator):
    """Generate the finite-length Ricker wavelet used by OpenFWI."""

    def __init__(self, **kwargs) -> None:
        temp = kwargs.get("freq", 15.0)
        if not isinstance(temp, (float, int)) or temp <= 0:
            raise ValueError("[freq] must be a positive float!")
        self.freq = float(temp)

        temp = kwargs.get("invert_polarity", True)
        if not isinstance(temp, bool):
            raise ValueError("[invert_polarity] must be a bool!")
        self.invert_polarity = temp

    def generate_amplitudes(
        self, nt: int, dt: float, offset_time: float, source_locations: torch.Tensor
    ) -> torch.Tensor:
        offset_nt = round(offset_time / dt)
        wavelet_nt = 2 * int((2.2 / self.freq / dt) // 2) + 1
        if offset_nt + wavelet_nt > nt:
            raise ValueError("The OpenFWI wavelet does not fit in the time series!")

        center = wavelet_nt // 2
        sample = torch.arange(
            1, wavelet_nt + 1, device=source_locations.device, dtype=torch.float32
        )
        beta = ((center - sample + 1) * self.freq * dt * torch.pi) ** 2
        wavelet = (1 - 2 * beta) * torch.exp(-beta)
        if self.invert_polarity:
            wavelet = -wavelet

        amplitudes = torch.zeros(
            (*source_locations.shape[:2], nt),
            device=source_locations.device,
            dtype=torch.float32,
        )
        amplitudes[:, :, offset_nt : offset_nt + wavelet_nt] = wavelet
        return amplitudes


@dataclass
class SourcesLocation:
    # 默认 sources_per_shot始终为 1
    # sources_per_shot: Optional[int] = field(
    #     metadata={"description": "每炮震源数"}
    # )
    source_depth: Optional[int] = field(
        metadata={"description": "震源深度 (网格索引)"}, default=None, init=True
    )
    source_spacing: Optional[int] = field(
        metadata={"description": "震源水平间距 (网格数),不提供则等分"},
        default=None,
        init=True,
    )
    first_source_offset: Optional[int] = field(
        metadata={
            "description": "首震源水平偏移 (网格数),避免误解，必须提供 source_spacing"
        },
        default=None,
        init=True,
    )
    sources_location: Optional[torch.Tensor] = field(
        metadata={"description": "自定义震源位置"}, default=None, init=True
    )

    def __post_init__(self):
        if self.sources_location is not None and (
            self.source_depth is not None
            or self.source_spacing is not None
            or self.first_source_offset is not None
        ):
            raise ValueError(
                "If source_locations is provided, source_depth, source_spacing, and first_source_offset cannot be provided."
            )
        elif (
            self.sources_location is None
            and self.first_source_offset is not None
            and self.source_spacing is None
        ):
            raise ValueError(
                "If first_source_offset is provided, source_spacing must be provided."
            )


@dataclass
class ReceiversLocation:
    receivers_per_shot: Optional[int] = field(
        metadata={"description": "每炮接收器数"}, default=None, init=True
    )
    receivers_depth: Optional[int] = field(
        metadata={"description": "接收器深度 (网格索引)"}, default=None, init=True
    )
    receivers_spacing: Optional[int] = field(
        metadata={"description": "接收器水平间距 (网格数),不提供则等分"},
        default=None,
        init=True,
    )
    first_receivers_offset: Optional[int] = field(
        metadata={
            "description": "首接收器水平偏移 (网格数),避免误解，必须提供 receivers_spacing"
        },
        default=None,
        init=True,
    )
    receivers_locations: Optional[torch.Tensor] = field(
        metadata={"description": "自定义接收器位置"}, default=None, init=True
    )

    def __post_init__(self):
        if self.receivers_locations is not None and (
            self.receivers_per_shot is not None
            or self.receivers_depth is not None
            or self.receivers_spacing is not None
            or self.first_receivers_offset is not None
        ):
            raise ValueError(
                "If receivers_locations is provided, receivers_per_shot, receivers_depth, receivers_spacing, and first_receivers_offset cannot be provided."
            )
        elif self.receivers_locations is None and self.receivers_per_shot is None:
            raise ValueError(
                "If receivers_locations is None, receivers_per_shot must be provided."
            )


class ForwardSimulator:
    def __init__(
        self,
        nz: int,
        nx: int,
        dx: float,
        dt: float,
        nt: int,
        shots: int,
        source_generator: AbstractSourceGenerator,
        sources_location: SourcesLocation,
        receivers_location: ReceiversLocation,
        **kwargs,
    ) -> None:
        """
        初始化地震波传播模拟器

        参数:
            nz: 深度方向网格数
            nx: 水平方向网格数
            dx: 空间采样间隔 (米)
            dt: 时间步长 (秒)
            nt: 时间步数
            shots: 炮点数量
            source_generator: 震源信号生成器对象
            sources_location: 震源位置配置对象
            receivers_location: 接收器位置配置对象
            **kwargs: 可选参数，包括:
                simulation_scale: 模拟比例因子，内部网格数 = 输入网格数 x scale
                pml_widths: PML 宽度 [top, bottom, left, right] 或 int, 0 表示刚性边界
                accuracy_order: 有限差分精度阶数 (2, 4, 6, 8)
                device: 计算设备 ['cpu', 'cuda']
                max_vel: 最大速度，用于 CFL 条件和 PML 构造
                pml_freq: PML 设计频率 (Hz)

        返回:
            None
        """
        self.nz: int = nz
        self.nx: int = nx
        self.dx: float = dx
        self.dt: float = dt
        self.nt: int = nt
        self.shots: int = shots
        self.sources_generator: AbstractSourceGenerator = source_generator
        self.sources_location: SourcesLocation = sources_location
        self.receivers_location: ReceiversLocation = receivers_location

        self.simulation_scale: float  # 模拟比例因子，内部网格数 = 输入网格数 × scale
        self.pml_widths: Optional[
            List[int]
        ]  # PML 宽度 [top, bottom, left, right], 0 表示刚性边界
        self.accuracy_order: int  # 有限差分精度阶数 (2, 4, 6, 8)
        self.device: str
        self.max_vel: Optional[float]  # 最大速度 (CFL 条件和 PML 构造用)
        self.pml_freq: float  # PML 设计频率 (Hz)
        self.offset_time: float

        self._sources_location: Optional[torch.Tensor]
        self._receivers_location: Optional[torch.Tensor]
        self._nz: Optional[int]  # simulation_scale with out pml
        self._nx: Optional[int]
        self._grid_spacing: Optional[float]
        self._vmodel: Optional[torch.Tensor]

        temp: Any

        # 处理模拟比例因子参数
        temp = kwargs.get("simulation_scale")
        if isinstance(temp, float):
            self.simulation_scale = temp
        elif temp is not None:
            raise ValueError("[simulation_scale] must be a float!")
        else:
            self.simulation_scale = 1.0

        # 处理 PML 宽度参数
        temp = kwargs.get("pml_widths")
        if isinstance(temp, list) or isinstance(temp, tuple):
            self.pml_widths = [int(i) for i in temp]
        elif isinstance(temp, int):
            self.pml_widths = [temp, temp, temp, temp]
        elif temp is not None:
            raise ValueError(
                "[pml_widths] must be a list or tuple with [top, bottom, left, right] or int!"
            )
        else:
            self.pml_widths = None

        # 处理有限差分精度阶数参数
        temp = kwargs.get("accuracy_order")
        if isinstance(temp, int) and temp in [2, 4, 6, 8]:
            self.accuracy_order = temp
        elif temp is not None:
            raise ValueError("[accuracy_order] must be one of [2, 4, 6, 8]!")
        else:
            self.accuracy_order = 4

        # 处理计算设备参数
        temp = kwargs.get("device")
        if isinstance(temp, str) and temp in ["cpu", "cuda"]:
            self.device = temp
        elif temp is not None:
            raise ValueError("[device] must be one of [cpu, cuda]!")
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        # 处理最大速度参数
        temp = kwargs.get("max_vel")
        if (isinstance(temp, float) or isinstance(temp, int)) and temp > 0:
            self.max_vel = float(temp)
        elif temp is not None:
            raise ValueError("[max_vel] must be a positive float!")
        else:
            self.max_vel = None

        # 处理 PML 设计频率参数
        temp = kwargs.get("pml_freq")
        if (isinstance(temp, float) or isinstance(temp, int)) and temp > 0:
            self.pml_freq = float(temp)
        elif temp is not None:
            raise ValueError("[pml_freq] must be a positive float!")
        else:
            self.pml_freq = 25.0

        temp = kwargs.get("offset_time")
        if (
            (isinstance(temp, float) or isinstance(temp, int))
            and temp > 0
            and temp < self.dt * self.nt
        ):
            self.offset_time = float(temp)
        elif temp is not None:
            raise ValueError("[offset_time] must be a float between 0 and dt * nt!")
        else:
            self.offset_time = 0.0

        self._check()

    def _check(self):
        """
        检查源位置和接收器位置的配置是否有效，包括边界检查和参数验证。

        该方法验证源位置和接收器位置的配置，确保它们在网格范围内且参数设置合理。
        验证包括：源位置的深度、间距、偏移量等参数，以及接收器位置的深度、
        每炮接收器数量、间距、偏移量等参数。
        """
        # 检查源位置配置
        if self.sources_location.sources_location is None:
            if self.sources_location.source_depth is None:
                self.sources_location.source_depth = 0
            else:
                if (
                    self.sources_location.source_depth < 0
                    or self.sources_location.source_depth >= self.nz
                ):
                    raise IndexError("[source_depth] is out of range!")
            if self.sources_location.source_spacing is not None:
                if self.sources_location.first_source_offset is None:
                    self.sources_location.first_source_offset = 0
                    if (
                        self.shots - 1
                    ) * self.sources_location.source_spacing >= self.nx:
                        raise IndexError("([shots] - 1) * [source_spacing] >= [nx]!")
                else:
                    if (
                        (self.shots - 1) * self.sources_location.source_spacing
                        + self.sources_location.first_source_offset
                        >= self.nx
                    ):
                        raise IndexError(
                            "([shots] - 1) * [source_spacing] + [first_source_offset] >= [nx]!"
                        )
        else:
            # 验证预定义源位置是否在网格范围内
            for i in range(self.sources_location.sources_location.shape[0]):
                for j in range(self.sources_location.sources_location.shape[1]):
                    if (
                        self.sources_location.sources_location[i, j][0] < 0
                        or self.sources_location.sources_location[i, j][0] >= self.nx
                        or self.sources_location.sources_location[i, j][1] < 0
                        or self.sources_location.sources_location[i, j][1] >= self.nz
                    ):
                        raise IndexError("[source_locations] is out of range!")

        # 检查接收器位置配置
        if self.receivers_location.receivers_locations is None:
            if self.receivers_location.receivers_depth is None:
                self.receivers_location.receivers_depth = 0
            else:
                if (
                    self.receivers_location.receivers_depth < 0
                    or self.receivers_location.receivers_depth >= self.nz
                ):
                    raise IndexError("[receivers_depth] is out of range!")
            assert self.receivers_location.receivers_per_shot is not None
            if self.receivers_location.receivers_per_shot > self.nx:
                raise IndexError("[receivers_per_shot] > [nx]!")
            if self.receivers_location.receivers_spacing is not None:
                if self.receivers_location.first_receivers_offset is None:
                    self.receivers_location.first_receivers_offset = 0
                    if (
                        self.receivers_location.receivers_per_shot - 1
                    ) * self.receivers_location.receivers_spacing >= self.nx:
                        raise IndexError(
                            "([receivers_per_shot] - 1) * [receivers_spacing] >= [nx]!"
                        )
                else:
                    if (
                        (self.receivers_location.receivers_per_shot - 1)
                        * self.receivers_location.receivers_spacing
                        + self.receivers_location.first_receivers_offset
                        >= self.nx
                    ):
                        raise IndexError(
                            "([receivers_per_shot] - 1) * [receivers_spacing] + [first_receivers_offset] >= [nx]!"
                        )
        else:
            # 验证预定义接收器位置是否在网格范围内
            for i in range(self.receivers_location.receivers_locations.shape[0]):
                for j in range(self.receivers_location.receivers_locations.shape[1]):
                    if (
                        self.receivers_location.receivers_locations[i, j][0] < 0
                        or self.receivers_location.receivers_locations[i, j][0]
                        >= self.nx
                        or self.receivers_location.receivers_locations[i, j][1] < 0
                        or self.receivers_location.receivers_locations[i, j][1]
                        >= self.nz
                    ):
                        raise IndexError("[receivers_locations] is out of range!")

        self._compute_location()

    def _compute_location(self):
        """
        计算仿真域的尺寸和位置信息，包括网格点数、PML 边界宽度以及源和接收器的位置。

        该方法根据仿真缩放因子调整网格尺寸，并计算 PML 边界宽度（如果存在），
        然后调用内部方法计算源和接收器的具体位置。

        参数:
            无参数

        返回值:
            无返回值
        """
        self._nz = int(self.nz * self.simulation_scale)
        self._nx = int(self.nx * self.simulation_scale)

        self._grid_spacing = self.dx / self.simulation_scale

        # 计算源和接收器的位置
        self._sources_location = self._compute_sources_location()
        self._receivers_location = self._compute_receivers_location()

    def _compute_sources_location(self) -> torch.Tensor:
        """
        计算震源位置坐标

        该函数根据震源配置信息计算震源在网格中的位置坐标。如果震源位置已预定义，则直接使用预定义位置；
        否则根据震源深度、间距等参数动态计算震源位置。震源位置为二维坐标 (x, z)，其中 x 为水平位置，z 为深度位置。

        Returns:
            torch.Tensor: 震源位置张量，形状为 (shots, 1, 2)，其中最后一维表示 (x, z) 坐标
        """
        if self.sources_location.sources_location is None:
            # 初始化震源位置张量，形状为 (shots, 1, 2)，初始值为 0
            locations: torch.Tensor = torch.zeros((self.shots, 1, 2), dtype=torch.long)
            assert self.sources_location.source_depth is not None
            # 设置震源深度坐标 (z 坐标)
            locations[:, :, 1] = round(
                self.sources_location.source_depth * self.simulation_scale
            )
            if self.sources_location.source_spacing is not None:
                # 当震源间距不为 None 时，根据间距和首震源偏移量计算震源 x 坐标
                assert self.sources_location.first_source_offset is not None
                locations[:, :, 0] = (
                    (
                        (
                            torch.arange(self.shots)
                            * self.sources_location.source_spacing
                            + self.sources_location.first_source_offset
                        )
                        * self.simulation_scale
                    )
                    .to(torch.long)
                    .view(self.shots, 1)
                )
            else:
                # 当震源间距为 None 时，均匀分布震源位置
                assert self._nx is not None
                locations[:, :, 0] = (
                    torch.round(torch.linspace(0, self._nx - 1, self.shots))
                    .to(torch.long)
                    .view(self.shots, 1)
                )
            return locations.cpu()
        else:
            # 当震源位置已预定义时，按比例缩放并转换为整数类型返回
            return (
                (self.simulation_scale * self.sources_location.sources_location)
                .to(torch.long).cpu()
            )

    def _compute_receivers_location(self) -> torch.Tensor:
        """
        计算接收器位置坐标

        该函数根据接收器配置参数计算每个接收器在每个炮点的位置坐标。
        如果已预设接收器位置，则直接使用预设位置；否则根据接收器参数动态计算位置。

        Returns:
            torch.Tensor: 接收器位置张量，形状为 (shots, receivers_per_shot, 2)，
                            其中最后一维表示 (x, z) 坐标
        """
        if self.receivers_location.receivers_locations is None:
            # 验证必要参数存在性
            assert self.receivers_location.receivers_per_shot is not None
            # 初始化位置张量，形状为 (炮点数，每炮接收器数，2)
            locations: torch.Tensor = torch.zeros(
                (self.shots, self.receivers_location.receivers_per_shot, 2),
                dtype=torch.long,
            )
            # 验证深度参数存在性
            assert self.receivers_location.receivers_depth is not None
            # 设置所有接收器的深度坐标
            locations[:, :, 1] = round(
                self.receivers_location.receivers_depth * self.simulation_scale
            )
            if self.receivers_location.receivers_spacing is not None:
                # 验证首接收器偏移量参数存在性
                assert self.receivers_location.first_receivers_offset is not None
                # 根据接收器间距和首接收器偏移量计算水平坐标
                locations[:, :, 0] = (
                    (
                        (
                            torch.arange(self.receivers_location.receivers_per_shot)
                            * self.receivers_location.receivers_spacing
                            + self.receivers_location.first_receivers_offset
                        )
                        * self.simulation_scale
                    )
                    .repeat(self.shots, 1)
                    .to(torch.long)
                )
            else:
                # 验证网格尺寸参数存在性
                assert self._nx is not None
                # 在网格范围内均匀分布接收器的水平坐标
                locations[:, :, 0] = (
                    torch.round(
                        torch.linspace(
                            0, self._nx - 1, self.receivers_location.receivers_per_shot
                        )
                    )
                    .repeat(self.shots, 1)
                    .to(torch.long)
                )
            return locations.cpu()
        else:
            # 使用预设的接收器位置并进行缩放转换
            return (
                (self.simulation_scale * self.receivers_location.receivers_locations)
                .to(torch.long).cpu()
            )

    def _compute_vmodel(self, vmodel: torch.Tensor):
        """
        计算并设置速度模型

        该方法将输入的速度模型张量插值到指定的尺寸，然后根据 PML 边界条件进行填充

        Args:
            vmodel (torch.Tensor): 输入的速度模型张量，需要被插值和处理

        Returns:
            None: 该方法直接修改实例变量 self._vmodel，无返回值
        """
        assert self._nz is not None
        assert self._nx is not None
        self._vmodel = (
            torch.nn.functional.interpolate(
                vmodel.unsqueeze(0).unsqueeze(0),
                size=(self._nx, self._nz),
                mode="bilinear",
                align_corners=False,
            )
            .to(torch.float)[0, 0]
            .to(self.device)
        )

        # Deepwave adds the PML outside this model. Padding here would create a
        # second boundary layer while leaving source/receiver coordinates unshifted.

    def _forward_check(self, v: torch.Tensor):
        """
        对输入张量进行前向检查和形状调整

        该函数验证输入张量的维度和形状是否符合要求，并根据需要进行转置操作。
        如果输入张量形状为 (nz, nx)，则转置为 (nx, nz)；如果已经是 (nx, nz) 则保持不变。
        当 nz==nx 时，无条件转置一次
        最终将张量转换为 float 类型。

        Args:
            v (torch.Tensor): 输入的 2D 张量，期望形状为 (self.nz, self.nx) 或 (self.nx, self.nz)

        Returns:
            torch.Tensor: 经过形状检查、可能转置和类型转换后的张量，形状为 (nx, nz)

        Raises:
            ValueError: 当输入张量不是 2D 张量或形状不符合要求时抛出异常
        """
        if len(v.shape) != 2:
            raise ValueError("v must be a 2D tensor")
        if not (v.shape[0] == self.nz and v.shape[1] == self.nx) and not (
            v.shape[0] == self.nx and v.shape[1] == self.nz
        ):
            raise ValueError("v must have shape ({}, {})".format(self._nz, self._nx))

        if self.nx == self.nz:
            warnings.warn(
                "The input tensor is a square tensor, please pay attention to the direction is [nz, nx](same to matplotlib)."
            )
        # 检查输入张量形状是否为 (nz, nx)，如果是则需要转置为 (nx, nz)
        if (
            v.shape[0] == self.nz and v.shape[1] == self.nx
        ):  # 输入张量是 (nz, nx),则变换为 (nx, nz)
            return v.transpose(0, 1).to(torch.float)
        else:
            return v.to(torch.float)

    def forward(
        self,
        v: torch.Tensor,
        sources_location: Optional[torch.Tensor] = None,
        receivers_location: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        执行前向传播计算，使用给定的速度张量计算地震波场模拟结果

        Args:
            v (torch.Tensor): 输入的速度张量，用于构建速度模型 (nz, nx)

        Returns:
            torch.Tensor: 地震波场模拟的输出结果（最后一个时间步的结果）
        """

        # 计算网格位置信息
        self._compute_location()
        assert self._grid_spacing is not None
        assert self._sources_location is not None
        assert self._receivers_location is not None

        src_loc: torch.Tensor = (
            sources_location.to(self.device)
            if sources_location is not None
            else self._sources_location.to(self.device)
        )
        rcv_loc: torch.Tensor = (
            receivers_location.to(self.device)
            if receivers_location is not None
            else self._receivers_location.to(self.device)
        )
        src_amps: torch.Tensor = self.sources_generator.generate_amplitudes(
            self.nt, self.dt, self.offset_time, src_loc
        ).to(self.device)
        if self.pml_widths is not None:
            dw_pml_width:Optional[list[int]] = [self.pml_widths[2], self.pml_widths[3], self.pml_widths[0], self.pml_widths[1]]
        else:
            dw_pml_width:Optional[list[int]] = None

        # 检查输入张量的有效性
        v = self._forward_check(v)
        # 基于输入速度张量计算速度模型
        self._compute_vmodel(v)
        assert self._vmodel is not None

        # 调用 deepwave 进行标量波场模拟计算
        # nx, nz
        out = deepwave.scalar(
            v=self._vmodel,
            grid_spacing=self._grid_spacing,
            dt=self.dt,
            source_amplitudes=src_amps,
            source_locations=src_loc,
            receiver_locations=rcv_loc,
            accuracy=self.accuracy_order,
            pml_width=dw_pml_width if dw_pml_width is not None else 20,
            pml_freq=self.pml_freq,
            max_vel=self.max_vel,
        )

        return out[-1].permute(0, 2, 1)


def create_openfwi_forward_simulator(
    device: Optional[str] = None,
    pml_width: int = 120,
    simulation_scale: float = 1.0,
) -> ForwardSimulator:
    """Create a Deepwave simulator matching the public 2D OpenFWI geometry.

    OpenFWI simulated 1001 steps but released arrays with 1000 time samples.
    Use :func:`simulate_openfwi` to apply the correct time-sample alignment.
    Deepwave's PML differs from OpenFWI's original damping-based ABC, so exact
    sample-by-sample equality with the released files is not expected.
    """
    kwargs = {
        "simulation_scale": float(simulation_scale),
        "pml_widths": pml_width,
        "accuracy_order": 4,
        "pml_freq": 15.0,
        "max_vel": 6000.0,
    }
    if device is not None:
        kwargs["device"] = device

    return ForwardSimulator(
        nz=70,
        nx=70,
        dx=10.0,
        dt=0.001,
        nt=1001,
        shots=5,
        source_generator=OpenFWIRickerSourceGenerator(freq=15.0),
        sources_location=SourcesLocation(source_depth=0),
        receivers_location=ReceiversLocation(
            receivers_per_shot=70,
            receivers_depth=0,
            receivers_spacing=1,
            first_receivers_offset=0,
        ),
        **kwargs,
    )


def simulate_openfwi(
    v: torch.Tensor, simulator: Optional[ForwardSimulator] = None
) -> torch.Tensor:
    """Simulate one OpenFWI velocity map and return shape ``[5, 1000, 70]``.

    Deepwave records before the current source sample affects the wavefield,
    while the OpenFWI solver records after source injection. Dropping the first
    Deepwave receiver sample aligns the two time conventions.
    """
    if simulator is None:
        simulator = create_openfwi_forward_simulator()
    if simulator.nt != 1001:
        raise ValueError("The OpenFWI simulator must use 1001 time steps!")
    return simulator.forward(v)[:, 1:1001]
