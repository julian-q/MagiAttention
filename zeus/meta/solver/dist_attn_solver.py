from dataclasses import dataclass

import torch.distributed as dist

from zeus.common.enum import AttnMaskType
from zeus.common.range import AttnRange
from zeus.common.ranges import AttnRanges

from ..collection.calc_meta import AttnArg, AttnCalcMeta
from ..collection.comm_meta import CommMeta, GroupCastCollectiveArg
from ..collection.dispatch_meta import DispatchMeta
from ..container.bucket import AttnBucket
from ..container.slice import AttnSlice


class AttnRangeWithRank(AttnRange):
    def __init__(self, rank_set: set[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_set = rank_set


class GroupCastRanges:
    def __init__(self, cp_size: int, ranges_list: list[AttnRanges]):
        assert len(ranges_list) == cp_size
        self.cp_size = cp_size
        self.ranges_list = ranges_list

        self.ranges_group = AttnRanges()

        for cp_rank, ranges in enumerate(ranges_list):
            for r in ranges:
                self.ranges_group.append(
                    AttnRangeWithRank(rank_set=set([cp_rank]), start=r.start, end=r.end)
                )

        self.ranges_group = self.ranges_group.sort()

    def split(self):
        self.ranges_group = self.ranges_group.sort()

        if len(self.ranges_group) <= 1:
            return

        new_ranges = AttnRanges()

        # 遍历所有可能的位置点来分割
        all_points = set()
        for r in self.ranges_group:
            all_points.add(r.start)
            all_points.add(r.end)

        all_points = sorted(list(all_points))

        # 对每个区间[p1,p2]判断它被哪些原始range覆盖
        for i in range(len(all_points) - 1):
            p1, p2 = all_points[i], all_points[i + 1]

            # 找出所有覆盖这个区间的ranges
            cover_rank_set = set()
            for r in self.ranges_group:
                r: AttnRangeWithRank
                if r.start <= p1 and r.end >= p2:
                    cover_rank_set.update(r.rank_set)

            if cover_rank_set:  # 如果有range覆盖这个区间
                new_ranges.append(
                    AttnRangeWithRank(rank_set=cover_rank_set, start=p1, end=p2)
                )

        self.ranges_group = new_ranges


@dataclass
class RankEntry:
    host_k_ranges_global: AttnRanges
    remote_k_ranges_global: AttnRanges
    attn_calc_host_slice_list: list[AttnSlice]
    attn_calc_remote_slice_list: list[AttnSlice]
    attn_calc_slice_list: list[AttnSlice]


@dataclass
class TableEntry:
    k_ranges_local_recv_from: AttnRanges
    k_ranges_local_send_to: AttnRanges
    k_ranges_global_transfer: AttnRanges


@dataclass
class TransferInfo:
    k_ranges_global_recv_from_each_rank_list: list[AttnRanges]
    k_ranges_local_recv_from_each_rank_list: list[AttnRanges]
    k_ranges_global_send_to_each_rank_list: list[AttnRanges]
    k_ranges_local_send_to_each_rank_list: list[AttnRanges]


class AttnSolver:
    def __init__(
        self,
        bucket_per_rank: list[AttnBucket],
        dispatch_meta_q: DispatchMeta,
        dispatch_meta_kv: DispatchMeta,
        cp_group_nccl: dist.ProcessGroup,
        cp_group_gloo: dist.ProcessGroup,
        overlap_degree: int,
    ):
        assert dist.get_backend(cp_group_nccl) == dist.Backend.NCCL
        assert dist.get_backend(cp_group_gloo) == dist.Backend.GLOO
        assert overlap_degree == 1

        self.cp_rank = dist.get_rank(cp_group_nccl)
        self.cp_size = dist.get_world_size(cp_group_nccl)

        self.dispatch_meta_q = dispatch_meta_q
        self.dispatch_meta_kv = dispatch_meta_kv
        self.cp_group_nccl = cp_group_nccl
        self.cp_group_gloo = cp_group_gloo

        self.bucket = bucket_per_rank[self.cp_rank]  # this rank

        # 计算在host上的q_ranges
        self.host_q_ranges_global = self.dispatch_meta_q.host_ranges_per_rank[
            self.cp_rank
        ].merge()
        assert self.host_q_ranges_global.is_merged()
        # 计算在host上的k_ranges和remote上的k_ranges
        self.host_k_ranges_global = self.dispatch_meta_kv.host_ranges_per_rank[
            self.cp_rank
        ].merge()  # this rank
        assert self.host_k_ranges_global.is_merged()
        self.remote_k_ranges_global = self.bucket.k_ranges.find_hole_ranges(
            self.host_k_ranges_global
        )
        assert self.remote_k_ranges_global.is_merged()
        (
            attn_calc_host_slice_list,
            attn_calc_remote_slice_list,
        ) = self.finish_attn_calc_qk_ranges(
            self.host_k_ranges_global,
            self.remote_k_ranges_global,
        )
        self.host_rank_entry = RankEntry(
            host_k_ranges_global=self.host_k_ranges_global,
            remote_k_ranges_global=self.remote_k_ranges_global,
            attn_calc_host_slice_list=attn_calc_host_slice_list,
            attn_calc_remote_slice_list=attn_calc_remote_slice_list,
            attn_calc_slice_list=self.bucket.attn_slices,
        )
        self.rank_entry_per_rank = self.finish_rank_entry_per_rank(self.host_rank_entry)
        self.table = self.finish_table()

    def finish_attn_calc_qk_ranges(
        self,
        host_k_ranges_global: AttnRanges,
        remote_k_ranges_global: AttnRanges,
    ):
        attn_calc_q_ranges_global = self.bucket.q_ranges
        attn_calc_k_ranges_global = self.bucket.k_ranges
        # 确保attn_q_ranges_global和attn_k_ranges_global的range数量相同
        assert len(attn_calc_q_ranges_global) == len(attn_calc_k_ranges_global)

        attn_calc_host_slice_list = []
        attn_calc_remote_slice_list = []
        for i, ith_attn_slice in enumerate(self.bucket.attn_slices):
            worker = AttnRanges()
            worker.append(ith_attn_slice.k_range)
            ith_attn_calc_host_k_ranges_global = worker.find_overlap_ranges(
                host_k_ranges_global
            )
            ith_attn_calc_remote_k_ranges_global = worker.find_hole_ranges(
                host_k_ranges_global
            )

            ith_attn_calc_host_k_ranges_local = host_k_ranges_global.make_ranges_local(
                ith_attn_calc_host_k_ranges_global
            ).merge()
            ith_attn_calc_remote_k_ranges_local = (
                remote_k_ranges_global.make_ranges_local(
                    ith_attn_calc_remote_k_ranges_global
                ).merge()
            )

            # attn的host_k_ranges和remote_k_ranges满足连续性, merge完成之后最多只剩一个range
            # 为0则代表没有k_ranges在host上或者在remote上
            assert (
                len(ith_attn_calc_host_k_ranges_local) <= 1
            ), f"{ith_attn_calc_host_k_ranges_local=}, {ith_attn_calc_host_k_ranges_global=}"
            assert (
                len(ith_attn_calc_remote_k_ranges_local) <= 1
            ), f"{ith_attn_calc_remote_k_ranges_local=}, {ith_attn_calc_remote_k_ranges_global=}"

            ith_attn_calc_host_q_range_local = (
                self.host_q_ranges_global.make_range_local(ith_attn_slice.q_range)
            )

            # HACK: 目前还没有考虑causal的setting, 因此全都是full-attn, 也就是ith_attn_slice.mask_type = AttnMaskType.FULL
            if len(ith_attn_calc_host_k_ranges_local) == 1:
                attn_calc_host_slice_list.append(
                    AttnSlice(
                        q_range=ith_attn_calc_host_q_range_local,
                        k_range=ith_attn_calc_host_k_ranges_local[0],
                        mask_type=ith_attn_slice.mask_type,
                    )
                )
            if len(ith_attn_calc_remote_k_ranges_local) == 1:
                attn_calc_remote_slice_list.append(
                    AttnSlice(
                        q_range=ith_attn_calc_host_q_range_local,
                        k_range=ith_attn_calc_remote_k_ranges_local[0],
                        mask_type=ith_attn_slice.mask_type,
                    )
                )

        return attn_calc_host_slice_list, attn_calc_remote_slice_list

    def finish_rank_entry_per_rank(self, host_rank_entry: RankEntry) -> list[RankEntry]:
        rank_entry_list = [None] * self.cp_size
        dist.all_gather_object(
            rank_entry_list, host_rank_entry, group=self.cp_group_gloo
        )

        return rank_entry_list  # type: ignore

    def finish_table(self):
        transfer_info = TransferInfo([], [], [], [])

        for rank in range(self.cp_size):
            if rank == self.cp_rank:
                transfer_info.k_ranges_global_recv_from_each_rank_list.append(
                    AttnRanges()
                )
                transfer_info.k_ranges_local_recv_from_each_rank_list.append(
                    AttnRanges()
                )
                continue

            rank_host_k_ranges_global = self.rank_entry_per_rank[
                rank
            ].host_k_ranges_global

            k_ranges_global_recv_from_rank = (
                self.remote_k_ranges_global.find_overlap_ranges(
                    rank_host_k_ranges_global
                )
            )
            transfer_info.k_ranges_global_recv_from_each_rank_list.append(
                k_ranges_global_recv_from_rank
            )
            k_ranges_local_recv_from_rank = (
                self.remote_k_ranges_global.make_ranges_local(
                    k_ranges_global_recv_from_rank
                )
            )
            transfer_info.k_ranges_local_recv_from_each_rank_list.append(
                k_ranges_local_recv_from_rank
            )

        for rank in range(self.cp_size):
            if rank == self.cp_rank:
                transfer_info.k_ranges_global_send_to_each_rank_list.append(
                    AttnRanges()
                )
                transfer_info.k_ranges_local_send_to_each_rank_list.append(AttnRanges())
                continue

            rank_remote_k_ranges_global = self.rank_entry_per_rank[
                rank
            ].remote_k_ranges_global

            k_ranges_global_send_to_rank = (
                self.host_k_ranges_global.find_overlap_ranges(
                    rank_remote_k_ranges_global
                )
            )
            transfer_info.k_ranges_global_send_to_each_rank_list.append(
                k_ranges_global_send_to_rank
            )
            k_ranges_local_send_to_rank = self.host_k_ranges_global.make_ranges_local(
                k_ranges_global_send_to_rank
            )
            transfer_info.k_ranges_local_send_to_each_rank_list.append(
                k_ranges_local_send_to_rank
            )

        transfer_info_per_rank: list[TransferInfo] = [None] * self.cp_size
        dist.all_gather_object(
            transfer_info_per_rank, transfer_info, group=self.cp_group_gloo
        )

        # sanity check
        for send_rank in range(self.cp_size):
            for recv_rank in range(self.cp_size):
                if send_rank == recv_rank:
                    continue
                send_info: TransferInfo = transfer_info_per_rank[send_rank]
                recv_info: TransferInfo = transfer_info_per_rank[recv_rank]
                k_ranges_global_recv_from_send_rank = (
                    recv_info.k_ranges_global_recv_from_each_rank_list[send_rank]
                )
                k_ranges_global_send_to_recv_rank = (
                    send_info.k_ranges_global_send_to_each_rank_list[recv_rank]
                )
                assert (
                    k_ranges_global_recv_from_send_rank
                    == k_ranges_global_send_to_recv_rank
                ), f"{send_rank=} {recv_rank=}\n{k_ranges_global_recv_from_send_rank=}\n{k_ranges_global_send_to_recv_rank=}"

        # 初始化 table
        table: list[list[TableEntry]] = []
        for send_rank in range(self.cp_size):
            table.append([])
            for recv_rank in range(self.cp_size):
                table[send_rank].append(
                    TableEntry(
                        k_ranges_local_recv_from=AttnRanges(),
                        k_ranges_local_send_to=AttnRanges(),
                        k_ranges_global_transfer=AttnRanges(),
                    )
                )

        for send_rank in range(self.cp_size):
            transfer_info = transfer_info_per_rank[send_rank]

            # 实例化group_cast_ranges_local_send_to
            group_cast_ranges_local_send_to = GroupCastRanges(
                cp_size=self.cp_size,
                ranges_list=transfer_info.k_ranges_local_send_to_each_rank_list,
            )
            group_cast_ranges_local_send_to.split()
            # 填充k_ranges_local_send_to
            for r in group_cast_ranges_local_send_to.ranges_group:
                r: AttnRangeWithRank
                for recv_rank in r.rank_set:
                    table[send_rank][recv_rank].k_ranges_local_send_to.append(
                        AttnRange(start=r.start, end=r.end)
                    )
            # 重排序
            for recv_rank in range(self.cp_size):
                table[send_rank][recv_rank].k_ranges_local_send_to = table[send_rank][
                    recv_rank
                ].k_ranges_local_send_to.sort()

            # 实例化group_cast_ranges_global_transfer
            group_cast_ranges_global_transfer = GroupCastRanges(
                cp_size=self.cp_size,
                ranges_list=transfer_info.k_ranges_global_send_to_each_rank_list,
            )
            group_cast_ranges_global_transfer.split()
            # 填充k_ranges_global_transfer
            for r in group_cast_ranges_global_transfer.ranges_group:
                r: AttnRangeWithRank
                for recv_rank in r.rank_set:
                    table[send_rank][recv_rank].k_ranges_global_transfer.append(
                        AttnRange(start=r.start, end=r.end)
                    )
            # 重排序
            for recv_rank in range(self.cp_size):
                table[send_rank][recv_rank].k_ranges_global_transfer = table[send_rank][
                    recv_rank
                ].k_ranges_global_transfer.sort()

            # 填充k_ranges_local_recv_from
            for recv_rank in range(self.cp_size):
                k_ranges_global_transfer = table[send_rank][
                    recv_rank
                ].k_ranges_global_transfer
                remote_k_ranges_global = self.rank_entry_per_rank[
                    recv_rank
                ].remote_k_ranges_global
                table[send_rank][
                    recv_rank
                ].k_ranges_local_recv_from = remote_k_ranges_global.make_ranges_local(
                    k_ranges_global_transfer
                )

        return table

    def calc_comm_meta(self) -> CommMeta:
        group_cast_ranges_local_send_to = GroupCastRanges(
            self.cp_size,
            [
                self.table[self.cp_rank][recv_rank].k_ranges_local_send_to
                for recv_rank in range(self.cp_size)
            ],
        )
        group_cast_ranges_local_send_to.split()

        input_split_size_list: list[int] = []
        dst_indices_list: list[list[int]] = []

        last_end = 0
        for r in group_cast_ranges_local_send_to.ranges_group:
            if r.start != last_end:
                input_split_size_list.append(r.start - last_end)
                dst_indices_list.append([])

            input_split_size_list.append(r.size)
            dst_indices_list.append(list(r.rank_set))  # type: ignore
            last_end = r.end

        if last_end != self.host_q_ranges_global.seqlen:
            input_split_size_list.append(self.host_q_ranges_global.seqlen - last_end)
            dst_indices_list.append([])

        # clac output args
        group_cast_ranges_local_recv_from = GroupCastRanges(
            self.cp_size,
            [
                self.table[send_rank][self.cp_rank].k_ranges_local_recv_from
                for send_rank in range(self.cp_size)
            ],
        )

        output_split_size_list = []
        src_index_list = []

        for recv_ranges in group_cast_ranges_local_recv_from.ranges_group:
            output_split_size_list.append(recv_ranges.size)
            assert len(recv_ranges.rank_set) == 1  # type: ignore
            src_index_list.append(recv_ranges.rank_set.pop())  # type: ignore

        comm_meta = CommMeta(
            num_remote_tokens_per_overlap_stage=[self.remote_k_ranges_global.seqlen],
            group_cast_collective_args_list=[
                GroupCastCollectiveArg(
                    input_split_size_list=input_split_size_list,
                    output_split_size_list=output_split_size_list,
                    dst_indices_list=dst_indices_list,
                    src_index_list=src_index_list,
                )
            ],
        )

        return comm_meta

    def calc_calc_meta(self) -> AttnCalcMeta:
        assert all(
            slice is not None
            for slice in self.host_rank_entry.attn_calc_host_slice_list
        )
        assert all(
            slice is not None
            for slice in self.host_rank_entry.attn_calc_remote_slice_list
        )

        attn_calc_meta = AttnCalcMeta(
            local_attn_arg=AttnArg(
                q_ranges=[
                    slice.q_range.to_naive_range()  # type: ignore
                    for slice in self.host_rank_entry.attn_calc_host_slice_list
                ],
                k_ranges=[
                    slice.k_range.to_naive_range()  # type: ignore
                    for slice in self.host_rank_entry.attn_calc_host_slice_list
                ],
                is_causal_mapping=[
                    slice.mask_type == AttnMaskType.CAUSAL
                    for slice in self.host_rank_entry.attn_calc_host_slice_list
                ],
            ),
            remote_attn_args_list=[
                AttnArg(
                    q_ranges=[
                        slice.q_range.to_naive_range()  # type: ignore
                        for slice in self.host_rank_entry.attn_calc_remote_slice_list
                    ],
                    k_ranges=[
                        slice.k_range.to_naive_range()  # type: ignore
                        for slice in self.host_rank_entry.attn_calc_remote_slice_list
                    ],
                    is_causal_mapping=[
                        slice.mask_type == AttnMaskType.CAUSAL
                        for slice in self.host_rank_entry.attn_calc_remote_slice_list
                    ],
                )
            ],
        )

        return attn_calc_meta

    def __repr__(self):
        # 计算每个单元格需要的最大宽度
        cell_widths = [[0] * self.cp_size for _ in range(self.cp_size)]
        for i in range(self.cp_size):
            for j in range(self.cp_size):
                send_str = f"send: {self.table[i][j].k_ranges_local_send_to}"
                recv_str = f"recv: {self.table[i][j].k_ranges_local_recv_from}"
                global_str = f"global: {self.table[i][j].k_ranges_global_transfer}"
                width = max(len(send_str), len(recv_str), len(global_str))
                cell_widths[i][j] = width

        # 计算每列的最大宽度
        col_widths = [
            max(
                max(cell_widths[i][j] for i in range(self.cp_size)),
                len(
                    f"host_k_ranges_global: {self.rank_entry_per_rank[j].host_k_ranges_global}"
                ),
                len(
                    f"remote_k_ranges_global: {self.rank_entry_per_rank[j].remote_k_ranges_global}"
                ),
                len(
                    f"attn_calc_host_slice_list: {self.rank_entry_per_rank[j].attn_calc_host_slice_list}"
                ),
                len(
                    f"attn_calc_remote_slice_list: {self.rank_entry_per_rank[j].attn_calc_remote_slice_list}"
                ),
                len(
                    f"attn_calc_slice_list: {self.rank_entry_per_rank[j].attn_calc_slice_list}"
                ),
            )
            for j in range(self.cp_size)
        ]

        # 计算表格的总宽度（考虑到每列分隔符 " | " 以及每行的"row xx |"前缀）
        table_width = (
            sum(col_widths) + 4 * (self.cp_size - 1) + 7
        )  # 每列间隔宽度4 + row xx | 前缀宽度为7

        # 构建表格
        result = []

        # 添加列标题行（扩展为5行高度）
        result.append("\n" + "-" * table_width)

        # 第一行：列号
        header_cells = [f"col{j:2d}".center(col_widths[j]) for j in range(self.cp_size)]
        result.append("r/c   | " + " | ".join(header_cells) + " |")

        # 第二行：host_k_ranges_global
        host_cells = [
            f"host_k_ranges_global: {self.rank_entry_per_rank[j].host_k_ranges_global}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        result.append("      | " + " | ".join(host_cells) + " |")

        # 第三行：remote_k_ranges_global
        remote_cells = [
            f"remote_k_ranges_global: {self.rank_entry_per_rank[j].remote_k_ranges_global}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        result.append("      | " + " | ".join(remote_cells) + " |")

        # 第四行：attn_calc_host_slice_list
        host_slice_cells = [
            f"attn_calc_host_slice_list: {self.rank_entry_per_rank[j].attn_calc_host_slice_list}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        result.append("      | " + " | ".join(host_slice_cells) + " |")

        # 第五行：attn_calc_remote_slice_list
        remote_slice_cells = [
            f"attn_calc_remote_slice_list: {self.rank_entry_per_rank[j].attn_calc_remote_slice_list}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        result.append("      | " + " | ".join(remote_slice_cells) + " |")

        # 第六行：attn_calc_slice_list
        attn_slice_cells = [
            f"attn_calc_slice_list: {self.rank_entry_per_rank[j].attn_calc_slice_list}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        result.append("      | " + " | ".join(attn_slice_cells) + " |")

        # 添加分割线
        result.append("-" * table_width)

        # 添加每一行
        for i in range(self.cp_size):
            # 处理每个单元格的三行内容
            cell_lines = []
            for j in range(self.cp_size):
                cell_content = [
                    f"send: {self.table[i][j].k_ranges_local_send_to}".ljust(
                        col_widths[j]
                    ),
                    f"recv: {self.table[i][j].k_ranges_local_recv_from}".ljust(
                        col_widths[j]
                    ),
                    f"global: {self.table[i][j].k_ranges_global_transfer}".ljust(
                        col_widths[j]
                    ),
                ]
                cell_lines.append(cell_content)

            # 组装每行的三行内容
            for line_idx in range(3):
                prefix = f"row{i:2d} |" if line_idx == 0 else "      |"
                line = [cell_lines[j][line_idx] for j in range(self.cp_size)]
                result.append(f"{prefix} " + " | ".join(line) + " |")

            result.append("-" * table_width)  # 每行后面添加分割线

        return "\n".join(result)
