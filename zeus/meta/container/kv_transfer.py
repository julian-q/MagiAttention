from dataclasses import dataclass
from typing import List, Literal, Set

import torch.distributed as dist

from zeus.common.range import AttnRange
from zeus.common.ranges import AttnRanges
from zeus.utils import nvtx


@dataclass
class RangeWithRank(AttnRange):
    def __init__(self, rank_set: Set[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_set = rank_set


class GroupCastRanges:
    def __init__(self, cp_size: int, ranges_list: List[AttnRanges]):
        assert len(ranges_list) == cp_size
        self.cp_size = cp_size
        self.ranges_list = ranges_list

        self.ranges_group = AttnRanges()

        for cp_rank, ranges in enumerate(ranges_list):
            for r in ranges:
                self.ranges_group.append(
                    RangeWithRank(rank_set=set([cp_rank]), start=r.start, end=r.end)
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
                r: RangeWithRank
                if r.start <= p1 and r.end >= p2:
                    cover_rank_set.update(r.rank_set)

            if cover_rank_set:  # 如果有range覆盖这个区间
                new_ranges.append(
                    RangeWithRank(rank_set=cover_rank_set, start=p1, end=p2)
                )

        self.ranges_group = new_ranges


class KVTransferTable:
    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        host_qk_ranges_global_per_rank: List[AttnRanges],
        remote_k_ranges_global_per_rank: List[AttnRanges],
    ):
        """
        初始化一个 cp_size x cp_size 的传输表格
        """
        self.cp_group = cp_group
        self.cp_size = dist.get_world_size(cp_group)
        self.cp_rank = dist.get_rank(cp_group)

        # 创建 cp_size x cp_size 的表格,每个元素是包含3个 AttnRanges 的列表
        self.table = [
            [
                {"send": AttnRanges(), "recv": AttnRanges(), "global": AttnRanges()}
                for _ in range(self.cp_size)
            ]
            for _ in range(self.cp_size)
        ]

        assert (
            len(host_qk_ranges_global_per_rank) == self.cp_size
        ), f"len(host_qk_ranges_global_per_rank) == {len(host_qk_ranges_global_per_rank)}, self.cp_size == {self.cp_size}"
        assert (
            len(remote_k_ranges_global_per_rank) == self.cp_size
        ), f"len(remote_k_ranges_global_per_rank) == {len(remote_k_ranges_global_per_rank)}, self.cp_size == {self.cp_size}"

        self.cp_size = self.cp_size
        self.host_qk_ranges_global_merged_per_rank: List[AttnRanges] = [
            host_qk_ranges_global.merge()
            for host_qk_ranges_global in host_qk_ranges_global_per_rank
        ]
        self.remote_k_ranges_global_merged_per_rank: List[AttnRanges] = [
            remote_k_ranges_global.merge()
            for remote_k_ranges_global in remote_k_ranges_global_per_rank
        ]

        self.input_buffer_size_list = [
            host_qk_ranges_global_per_rank[cp_rank].seqlen
            for cp_rank in range(self.cp_size)
        ]
        self.output_buffer_size_list = [
            remote_k_ranges_global_per_rank[cp_rank].seqlen
            for cp_rank in range(self.cp_size)
        ]

    def get_ranges(
        self, i: int, j: int, type: Literal["send", "recv", "global"]
    ) -> AttnRanges:
        """
        获取表格中指定位置的注意力范围列表

        Args:
            i: 行索引
            j: 列索引

        Returns:
            包含3个元素的 AttnRanges 列表:
            - [0]: local_range_for_send
            - [1]: local_range_for_recv
            - [2]: global_range
        """
        if type not in ["send", "recv", "global"]:
            raise ValueError(f"Invalid type: {type}")

        assert 0 <= i < self.cp_size
        assert 0 <= j < self.cp_size

        if type == "send":
            return self.table[i][j]["send"]
        elif type == "recv":
            return self.table[i][j]["recv"]
        elif type == "global":
            return self.table[i][j]["global"]
        else:
            pass

    @nvtx.instrument_nvtx
    def correct(self):
        # TODO(xiaowu): 这个函数可以移到AttnRanges中
        # def is_sorted(lst):
        #     return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

        # for i in range(self.cp_size):
        #     for j in range(self.cp_size):
        #         send_ranges = self.get_ranges(i, j, "send")
        #         recv_ranges = self.get_ranges(i, j, "recv")
        #         global_ranges = self.get_ranges(i, j, "global")
        #         assert is_sorted(send_ranges)
        #         assert is_sorted(recv_ranges)
        #         assert is_sorted(global_ranges)

        # finish table using communication
        if dist.get_backend(self.cp_group) == dist.Backend.GLOO:
            recv_ranges_list_global = [None] * self.cp_size
            recv_ranges_list_this_rank = [
                self.table[i][self.cp_rank]["recv"] for i in range(self.cp_size)
            ]
            dist.all_gather_object(
                recv_ranges_list_global, recv_ranges_list_this_rank, group=self.cp_group
            )

            for send_rank in range(self.cp_size):
                for recv_rank in range(self.cp_size):
                    self.table[send_rank][recv_rank]["recv"] = recv_ranges_list_global[
                        recv_rank
                    ][send_rank]

            global_ranges_list_global = [None] * self.cp_size
            global_ranges_list_this_rank = [
                self.table[i][self.cp_rank]["global"] for i in range(self.cp_size)
            ]
            dist.all_gather_object(
                global_ranges_list_global,
                global_ranges_list_this_rank,
                group=self.cp_group,
            )

            for send_rank in range(self.cp_size):
                for recv_rank in range(self.cp_size):
                    self.table[send_rank][recv_rank][
                        "global"
                    ] = global_ranges_list_global[recv_rank][send_rank]

            send_ranges_list_global = [None] * self.cp_size
            send_ranges_list_this_rank = [
                self.table[self.cp_rank][i]["send"] for i in range(self.cp_size)
            ]
            dist.all_gather_object(
                send_ranges_list_global, send_ranges_list_this_rank, group=self.cp_group
            )

            for send_rank in range(self.cp_size):
                for recv_rank in range(self.cp_size):
                    self.table[send_rank][recv_rank]["send"] = send_ranges_list_global[
                        send_rank
                    ][recv_rank]

        else:
            raise NotImplementedError(
                f"Backend {dist.get_backend(self.cp_group)} is not supported"
            )

        for i in range(self.cp_size):
            gcr_send = GroupCastRanges(
                self.cp_size,
                [self.get_ranges(i, j, "send") for j in range(self.cp_size)],
            )
            for j in range(self.cp_size):
                self.table[i][j]["send"] = AttnRanges()
            gcr_send.split()
            for r in gcr_send.ranges_group:
                r: RangeWithRank
                for rank in r.rank_set:
                    self.table[i][rank]["send"].append(
                        AttnRange(start=r.start, end=r.end)
                    )
            for j in range(self.cp_size):
                self.table[i][j]["send"] = self.table[i][j]["send"].sort()

            gcr_global = GroupCastRanges(
                self.cp_size,
                [self.get_ranges(i, j, "global") for j in range(self.cp_size)],
            )
            for j in range(self.cp_size):
                self.table[i][j]["global"] = AttnRanges()
            gcr_global.split()
            for r in gcr_global.ranges_group:
                r: RangeWithRank
                for rank in r.rank_set:
                    self.table[i][rank]["global"].append(
                        AttnRange(start=r.start, end=r.end)
                    )
            for j in range(self.cp_size):
                self.table[i][j]["global"] = self.table[i][j]["global"].sort()

            for j in range(self.cp_size):
                global_ranges = self.table[i][j]["global"]
                self.table[i][j]["recv"] = self.remote_k_ranges_global_merged_per_rank[
                    j
                ].make_ranges_local(global_ranges, merged=False)
                self.table[i][j]["recv"] = self.table[i][j]["recv"].sort()

    def to_group_cast_args(
        self, cp_rank: int
    ) -> tuple[list[int], list[list[int]], list[int], list[int]]:
        # calc input args
        send_ranges_list = [
            self.get_ranges(cp_rank, j, "send") for j in range(self.cp_size)
        ]
        gcr_send = GroupCastRanges(self.cp_size, send_ranges_list)
        gcr_send.split()

        input_split_size_list: list[int] = []
        dst_indices_list: list[list[int]] = []

        last_end = 0
        for r in gcr_send.ranges_group:
            if r.start != last_end:
                input_split_size_list.append(r.start - last_end)
                dst_indices_list.append([])

            input_split_size_list.append(r.size)
            dst_indices_list.append(list(r.rank_set))  # type: ignore
            last_end = r.end

        if last_end != self.input_buffer_size_list[cp_rank]:
            input_split_size_list.append(
                self.input_buffer_size_list[cp_rank] - last_end
            )
            dst_indices_list.append([])

        # clac output args
        recv_ranges_list = [
            self.get_ranges(j, cp_rank, "recv") for j in range(self.cp_size)
        ]
        gcr_recv = GroupCastRanges(self.cp_size, recv_ranges_list)

        output_split_size_list = []
        src_index_list = []

        for recv_ranges in gcr_recv.ranges_group:
            output_split_size_list.append(recv_ranges.size)
            assert len(recv_ranges.rank_set) == 1  # type: ignore
            src_index_list.append(recv_ranges.rank_set.pop())  # type: ignore

        return (
            input_split_size_list,
            dst_indices_list,
            output_split_size_list,
            src_index_list,
        )

    def __repr__(self):
        # 计算每个单元格需要的最大宽度
        cell_widths = [[0] * self.cp_size for _ in range(self.cp_size)]
        for i in range(self.cp_size):
            for j in range(self.cp_size):
                send_str = f"send: {self.table[i][j]['send']}"
                recv_str = f"recv: {self.table[i][j]['recv']}"
                global_str = f"global: {self.table[i][j]['global']}"
                width = max(len(send_str), len(recv_str), len(global_str))
                cell_widths[i][j] = width

        # 计算每列的最大宽度
        col_widths = [
            max(cell_widths[i][j] for i in range(self.cp_size))
            for j in range(self.cp_size)
        ]

        # 计算表格的总宽度（考虑到每列分隔符 " | " 以及每行的"row xx |"前缀）
        table_width = (
            sum(col_widths) + 4 * (self.cp_size - 1) + 7
        )  # 每列间隔宽度4 + row xx | 前缀宽度为7

        # 构建表格
        result = []

        # 添加列标题行
        header_cells = [f"col{j:2d}".center(col_widths[j]) for j in range(self.cp_size)]
        result.append("\n" + "-" * table_width)
        header = "r/c   | " + " | ".join(header_cells) + " |"
        result.append(header)

        # 添加分割线
        result.append("-" * table_width)

        # 添加每一行
        for i in range(self.cp_size):
            # 处理每个单元格的三行内容
            cell_lines = []
            for j in range(self.cp_size):
                cell_content = [
                    f"send: {self.table[i][j]['send']}".ljust(col_widths[j]),
                    f"recv: {self.table[i][j]['recv']}".ljust(col_widths[j]),
                    f"global: {self.table[i][j]['global']}".ljust(col_widths[j]),
                ]
                cell_lines.append(cell_content)

            # 组装每行的三行内容
            for line_idx in range(3):
                prefix = f"row{i:2d} |" if line_idx == 0 else "      |"
                line = [cell_lines[j][line_idx] for j in range(self.cp_size)]
                result.append(f"{prefix} " + " | ".join(line) + " |")

            result.append("-" * table_width)  # 每行后面添加分割线

        return "\n".join(result)
