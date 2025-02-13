from dataclasses import dataclass

import torch.distributed as dist

from zeus.common.enum import AttnMaskType, AttnOverlapMode
from zeus.common.range import AttnRange
from zeus.common.ranges import AttnRanges
from zeus.meta.collection.calc_meta import AttnArg, AttnCalcMeta
from zeus.meta.collection.comm_meta import CommMeta, GroupCastCollectiveArg
from zeus.meta.collection.dispatch_meta import DispatchMeta
from zeus.meta.container.bucket import AttnBucket
from zeus.meta.container.slice import AttnSlice
from zeus.utils import nvtx, transpose_matrix


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
                    AttnRangeWithRank(rank_set={cp_rank}, start=r.start, end=r.end)
                )

        self.ranges_group = self.ranges_group.sort()

    def split(self):
        self.ranges_group = self.ranges_group.sort()

        if len(self.ranges_group) <= 1:
            return

        new_ranges_group = AttnRanges()

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
                new_ranges_group.append(
                    AttnRangeWithRank(rank_set=cover_rank_set, start=p1, end=p2)
                )

        self.ranges_group = new_ranges_group


@dataclass
class HostRankEntry:
    host_q_ranges_global: AttnRanges
    host_k_ranges_global: AttnRanges
    remote_k_ranges_global: AttnRanges

    attn_calc_slice_global_list: list[AttnSlice]
    attn_calc_host_slice_local_list: list[AttnSlice]
    attn_calc_remote_slice_local_list: list[AttnSlice]

    # HACK: for multi-stage overlap
    attn_calc_remote_k_ranges_global_list: list[AttnRanges]


@dataclass
class RemoteRankEntry:
    host_k_ranges_global: AttnRanges
    remote_k_ranges_global: AttnRanges

    attn_calc_remote_slice_local_list: list[AttnSlice]


@dataclass
class TransferInfo:
    k_ranges_global_recv_from_each_rank_list: list[AttnRanges]
    k_ranges_local_recv_from_each_rank_list: list[AttnRanges]
    k_ranges_global_send_to_each_rank_list: list[AttnRanges]
    k_ranges_local_send_to_each_rank_list: list[AttnRanges]


@dataclass
class TableEntry:
    """The entry dataclass for transfer table,
    where:
        1. k_ranges_global: global k ranges to send w.r.t. to send rank's dispatch meta
        2. k_ranges_local_in_send_buf: local k ranges to send w.r.t. to send rank's send buf
        3. k_ranges_local_in_recv_buf: local k ranges to send w.r.t. to recv rank's recv buf
    """

    k_ranges_global: AttnRanges
    k_ranges_local_in_send_buf: AttnRanges
    k_ranges_local_in_recv_buf: AttnRanges


class TransferTable:
    """The transfer table class, maintaining [cp_size, cp_size] entries,
    where table[send_rank][recv_rank] is the send entry from send_rank to recv_rank

    Therefore:
        1. we can get the send args for group cast collective
            using 'k_ranges_local_in_send_buf' in the row of table[this_rank][...]
        2. we can get the recv args for group cast collective
            using 'k_ranges_local_in_recv_buf' in the column of table[...][this_rank]
    """

    def __init__(self, cp_size: int):
        self.cp_size = cp_size
        self._transfer_table: list[list[TableEntry]] = []

        # init each entry in the transfer table
        for send_rank in range(cp_size):
            self._transfer_table.append([])
            for recv_rank in range(cp_size):
                self._transfer_table[send_rank].append(
                    TableEntry(
                        k_ranges_global=AttnRanges(),
                        k_ranges_local_in_send_buf=AttnRanges(),
                        k_ranges_local_in_recv_buf=AttnRanges(),
                    )
                )

    # get
    def get_k_ranges_global(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> AttnRanges:
        return self._transfer_table[send_rank][recv_rank].k_ranges_global

    def get_k_ranges_local_in_send_buf(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> AttnRanges:
        return self._transfer_table[send_rank][recv_rank].k_ranges_local_in_send_buf

    def get_k_ranges_local_in_recv_buf(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> AttnRanges:
        return self._transfer_table[send_rank][recv_rank].k_ranges_local_in_recv_buf

    # append
    def append_k_ranges_global(
        self,
        send_rank: int,
        recv_rank: int,
        k_range: AttnRange,
    ) -> None:
        self._transfer_table[send_rank][recv_rank].k_ranges_global.append(k_range)

    def append_k_ranges_local_in_send_buf(
        self,
        send_rank: int,
        recv_rank: int,
        k_range: AttnRange,
    ) -> None:
        self._transfer_table[send_rank][recv_rank].k_ranges_local_in_send_buf.append(
            k_range
        )

    # sort
    def sort_k_ranges_global(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> None:
        self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_global = self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_global.sort()

    def sort_k_ranges_local_in_send_buf(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> None:
        self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_local_in_send_buf = self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_local_in_send_buf.sort()

    # make
    def make_k_ranges_local_in_recv_buf(
        self,
        send_rank: int,
        recv_rank: int,
        remote_k_ranges_global_for_recv_rank: AttnRanges,
    ) -> None:
        """Construct local k_ranges w.r.t. recv rank's recv buffer
        from host global k_ranges to send from send_rank to recv_rank
        and remote global k_ranges to recv from send_rank to recv_rank

        NOTE: this is the special attribute that should NOT be passed in from outside,
        but ONLY constructed internally
        """
        self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_local_in_recv_buf = remote_k_ranges_global_for_recv_rank.make_ranges_local(
            self._transfer_table[send_rank][recv_rank].k_ranges_global
        )


class AttnSolver:
    def __init__(
        self,
        bucket_per_rank: list[AttnBucket],
        dispatch_meta_q: DispatchMeta,
        dispatch_meta_kv: DispatchMeta,
        cp_group_nccl: dist.ProcessGroup,
        cp_group_gloo: dist.ProcessGroup,
        overlap_mode: AttnOverlapMode,
        overlap_degree: int | None,
    ):
        assert dist.get_backend(cp_group_nccl) == dist.Backend.NCCL
        assert dist.get_backend(cp_group_gloo) == dist.Backend.GLOO

        # TODO: limited to static overlap mode with overlap degree = 1 for now
        assert (
            overlap_mode is AttnOverlapMode.STATIC and overlap_degree == 1
        ), "For now, only supports static overlap mode with overlap degree 1."

        self.cp_rank = dist.get_rank(cp_group_nccl)
        self.cp_size = dist.get_world_size(cp_group_nccl)

        self.cp_group_nccl = cp_group_nccl
        self.cp_group_gloo = cp_group_gloo

        self.overlap_mode = overlap_mode
        # NOTE: this is the initial overlap degree,
        # which might be changed after overlap solver if overlap mode is dynamic
        self.overlap_degree = overlap_degree

        bucket_this_rank = bucket_per_rank[self.cp_rank]

        # init host q_ranges global for this rank
        host_q_ranges_global_this_rank = dispatch_meta_q.host_ranges_per_rank[
            self.cp_rank
        ].merge()
        assert host_q_ranges_global_this_rank.is_merged()

        # init host k_ranges global for this rank
        host_k_ranges_global_this_rank = dispatch_meta_kv.host_ranges_per_rank[
            self.cp_rank
        ].merge()
        assert host_k_ranges_global_this_rank.is_merged()

        # init remote k_ranges global for this rank
        remote_k_ranges_global_this_rank = bucket_this_rank.k_ranges.find_hole_ranges(
            host_k_ranges_global_this_rank
        )
        assert remote_k_ranges_global_this_rank.is_merged()

        # init host rank entry for this rank
        self.host_rank_entry_this_rank = self._init_host_rank_entry_this_rank(
            host_q_ranges_global=host_q_ranges_global_this_rank,
            host_k_ranges_global=host_k_ranges_global_this_rank,
            remote_k_ranges_global=remote_k_ranges_global_this_rank,
            attn_calc_q_ranges_global=bucket_this_rank.q_ranges,
            attn_calc_k_ranges_global=bucket_this_rank.k_ranges,
            attn_calc_slice_global_list=bucket_this_rank.attn_slices,
        )

        # init remote rank entry for each stage for this rank
        self.remote_rank_entry_per_stage_this_rank = (
            self._init_remote_rank_entry_per_stage_this_rank(
                self.host_rank_entry_this_rank
            )
        )

        # init remote rank entry for each rank for each stage
        self.remote_rank_entry_per_rank_per_stage = (
            self._init_remote_rank_entry_per_rank_per_stage(
                self.remote_rank_entry_per_stage_this_rank
            )
        )

        # init transfer table per stage
        self.transfer_table_per_stage: list[TransferTable] = []
        for stage, remote_rank_entry_per_rank_this_stage in enumerate(
            self.remote_rank_entry_per_rank_per_stage
        ):
            self.transfer_table_per_stage.append(
                self._init_transfer_table_for_one_stage(
                    stage,
                    remote_rank_entry_per_rank_this_stage,
                )
            )

    @nvtx.instrument_nvtx
    def _init_host_rank_entry_this_rank(
        self,
        host_q_ranges_global: AttnRanges,
        host_k_ranges_global: AttnRanges,
        remote_k_ranges_global: AttnRanges,
        attn_calc_q_ranges_global: AttnRanges,
        attn_calc_k_ranges_global: AttnRanges,
        attn_calc_slice_global_list: list[AttnSlice],
    ) -> HostRankEntry:
        assert len(attn_calc_q_ranges_global) == len(attn_calc_k_ranges_global), (
            f"The {len(attn_calc_q_ranges_global)=} should be equal to "
            f"{len(attn_calc_k_ranges_global)=}."
        )

        attn_calc_host_slice_local_list = []
        attn_calc_remote_slice_local_list = []
        attn_calc_remote_k_ranges_global_list = []
        for ith_attn_slice_global in attn_calc_slice_global_list:
            worker = AttnRanges()
            worker.append(ith_attn_slice_global.k_range)  # type: ignore
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

            ith_attn_calc_host_q_range_local = host_q_ranges_global.make_range_local(
                ith_attn_slice_global.q_range
            )

            # HACK: 目前还没有考虑causal的setting, 因此全都是full-attn,
            # 也就是ith_attn_slice.mask_type = AttnMaskType.FULL
            if len(ith_attn_calc_host_k_ranges_local) == 1:
                attn_calc_host_slice_local_list.append(
                    AttnSlice(
                        q_range=ith_attn_calc_host_q_range_local,
                        k_range=ith_attn_calc_host_k_ranges_local[0],
                        mask_type=ith_attn_slice_global.mask_type,
                    )
                )
            if len(ith_attn_calc_remote_k_ranges_local) == 1:
                attn_calc_remote_slice_local_list.append(
                    AttnSlice(
                        q_range=ith_attn_calc_host_q_range_local,
                        k_range=ith_attn_calc_remote_k_ranges_local[0],
                        mask_type=ith_attn_slice_global.mask_type,
                    )
                )
                attn_calc_remote_k_ranges_global_list.append(
                    ith_attn_calc_remote_k_ranges_global
                )

        host_rank_entry_this_rank = HostRankEntry(
            host_q_ranges_global=host_q_ranges_global,
            host_k_ranges_global=host_k_ranges_global,
            remote_k_ranges_global=remote_k_ranges_global,
            attn_calc_slice_global_list=attn_calc_slice_global_list,
            attn_calc_host_slice_local_list=attn_calc_host_slice_local_list,
            attn_calc_remote_slice_local_list=attn_calc_remote_slice_local_list,
            attn_calc_remote_k_ranges_global_list=attn_calc_remote_k_ranges_global_list,
        )

        return host_rank_entry_this_rank

    @nvtx.instrument_nvtx
    def _init_remote_rank_entry_per_stage_this_rank(
        self,
        host_rank_entry_this_rank: HostRankEntry,
    ) -> list[RemoteRankEntry]:
        if self.overlap_mode is AttnOverlapMode.STATIC and self.overlap_degree == 1:
            # HACK: for now, only support static mode with overlap degree 1,
            # which constructs remote rank entry just by copying from host rank entry
            remote_rank_entry_per_stage_this_rank = [
                RemoteRankEntry(
                    host_k_ranges_global=host_rank_entry_this_rank.host_k_ranges_global,
                    remote_k_ranges_global=host_rank_entry_this_rank.remote_k_ranges_global,
                    attn_calc_remote_slice_local_list=host_rank_entry_this_rank.attn_calc_remote_slice_local_list,
                )
            ]
        else:
            raise NotImplementedError("TODO: support overlap solver")

        return remote_rank_entry_per_stage_this_rank

    def _init_remote_rank_entry_per_rank_per_stage(
        self, remote_rank_entry_per_stage_this_rank: list[RemoteRankEntry]
    ) -> list[list[RemoteRankEntry]]:
        # all gather remote rank entry per stage from each rank
        remote_rank_entry_per_stage_per_rank = [None] * self.cp_size
        dist.all_gather_object(
            remote_rank_entry_per_stage_per_rank,
            remote_rank_entry_per_stage_this_rank,
            group=self.cp_group_nccl,
        )

        # check shape to be [cp_size, overlap_degree]
        assert (
            len(remote_rank_entry_per_stage_per_rank) == self.cp_size
            and len(remote_rank_entry_per_stage_per_rank[0]) == self.overlap_degree  # type: ignore
        )

        # transpose to be remote rank entry per rank for each stage
        remote_rank_entry_per_rank_per_stage = transpose_matrix(
            remote_rank_entry_per_stage_per_rank  # type: ignore
        )

        # check shape to be [overlap_degree, cp_size]
        assert (
            len(remote_rank_entry_per_rank_per_stage) == self.overlap_degree
            and len(remote_rank_entry_per_rank_per_stage[0]) == self.cp_size
        )

        return remote_rank_entry_per_rank_per_stage

    @nvtx.instrument_nvtx
    def _init_transfer_table_for_one_stage(
        self,
        stage: int,
        remote_rank_entry_per_rank_this_stage: list[RemoteRankEntry],
    ) -> TransferTable:
        # init transfer info for this rank
        transfer_info_this_rank = self._init_transfer_info_this_rank(
            remote_rank_entry_per_rank_this_stage
        )

        # init transfer info for each rank
        transfer_info_per_rank = self._init_transfer_info_per_rank(
            stage, transfer_info_this_rank
        )

        # init transfer table entry for each rank pair: (send_ranki, recv_rankj)
        transfer_table = TransferTable(cp_size=self.cp_size)

        # fill up transfer table
        for send_rank in range(self.cp_size):  # for each send_ranki
            transfer_info = transfer_info_per_rank[send_rank]

            # init group_cast_ranges for local k ranges that send_ranki needs to send to
            group_cast_ranges_local_send_to = GroupCastRanges(
                cp_size=self.cp_size,
                ranges_list=transfer_info.k_ranges_local_send_to_each_rank_list,
            )

            # split the local ranges into non-overlapped local ranges
            group_cast_ranges_local_send_to.split()

            # for each non-overlapped local k range that send_ranki needs to send to
            # we tranverse each dest recv_rankj to recv it in the set,
            # and append it to k_ranges_local_in_send_buf at the (send_ranki, recv_rankj) table entry
            for r in group_cast_ranges_local_send_to.ranges_group:
                r: AttnRangeWithRank  # type: ignore
                for recv_rank in r.rank_set:  # type: ignore
                    transfer_table.append_k_ranges_local_in_send_buf(
                        send_rank=send_rank,
                        recv_rank=recv_rank,
                        k_range=AttnRange(start=r.start, end=r.end),
                    )

            # sort the local k ranges to send for each dest recv_rankj
            for recv_rank in range(self.cp_size):
                transfer_table.sort_k_ranges_local_in_send_buf(
                    send_rank=send_rank,
                    recv_rank=recv_rank,
                )

            # init group_cast_ranges for global k ranges that send_ranki needs to send to
            group_cast_ranges_global_transfer = GroupCastRanges(
                cp_size=self.cp_size,
                ranges_list=transfer_info.k_ranges_global_send_to_each_rank_list,
            )

            # split the global ranges into non-overlapped global ranges
            group_cast_ranges_global_transfer.split()

            # for each non-overlapped global k range that send_ranki needs to send to
            # we tranverse each dest recv_rankj to recv it in the set,
            # and append it to k_ranges_local_in_send_buf at the (send_ranki, recv_rankj) table entry
            for r in group_cast_ranges_global_transfer.ranges_group:
                r: AttnRangeWithRank  # type: ignore
                for recv_rank in r.rank_set:  # type: ignore
                    transfer_table.append_k_ranges_global(
                        send_rank=send_rank,
                        recv_rank=recv_rank,
                        k_range=AttnRange(start=r.start, end=r.end),
                    )

            # sort the global k ranges to send for each dest recv_rankj
            for recv_rank in range(self.cp_size):
                transfer_table.sort_k_ranges_global(
                    send_rank=send_rank,
                    recv_rank=recv_rank,
                )

            # fill k_ranges_local_in_recv_buf
            for recv_rank in range(self.cp_size):
                remote_k_ranges_global_for_recv_rank = (
                    remote_rank_entry_per_rank_this_stage[
                        recv_rank
                    ].remote_k_ranges_global
                )

                transfer_table.make_k_ranges_local_in_recv_buf(
                    send_rank,
                    recv_rank,
                    remote_k_ranges_global_for_recv_rank,
                )

        return transfer_table

    @nvtx.instrument_nvtx
    def _init_transfer_info_this_rank(
        self,
        remote_rank_entry_per_rank_this_stage: list[RemoteRankEntry],
    ) -> TransferInfo:
        host_k_ranges_global_this_rank = remote_rank_entry_per_rank_this_stage[
            self.cp_rank
        ].host_k_ranges_global
        remote_k_ranges_global_this_rank = remote_rank_entry_per_rank_this_stage[
            self.cp_rank
        ].remote_k_ranges_global

        transfer_info_this_rank = TransferInfo([], [], [], [])

        # 1. initalize recv transfer info
        for rank in range(self.cp_size):
            if rank == self.cp_rank:  # no need to recv from this rank
                transfer_info_this_rank.k_ranges_global_recv_from_each_rank_list.append(
                    AttnRanges()
                )
                transfer_info_this_rank.k_ranges_local_recv_from_each_rank_list.append(
                    AttnRanges()
                )
                continue

            # get the global k ranges that this rank needs to recv from current rank
            rank_host_k_ranges_global = remote_rank_entry_per_rank_this_stage[
                rank
            ].host_k_ranges_global
            k_ranges_global_recv_from_rank = (
                remote_k_ranges_global_this_rank.find_overlap_ranges(
                    rank_host_k_ranges_global
                )
            )
            # make the global k ranges local w.r.t. self's recv buffer
            k_ranges_local_recv_from_rank = (
                remote_k_ranges_global_this_rank.make_ranges_local(
                    k_ranges_global_recv_from_rank
                )
            )
            # add to recv transfer info for both global and local ones
            transfer_info_this_rank.k_ranges_global_recv_from_each_rank_list.append(
                k_ranges_global_recv_from_rank
            )
            transfer_info_this_rank.k_ranges_local_recv_from_each_rank_list.append(
                k_ranges_local_recv_from_rank
            )

        # 2. initalize send transfer info
        for rank in range(self.cp_size):
            if rank == self.cp_rank:  # no need to send to this rank
                transfer_info_this_rank.k_ranges_global_send_to_each_rank_list.append(
                    AttnRanges()
                )
                transfer_info_this_rank.k_ranges_local_send_to_each_rank_list.append(
                    AttnRanges()
                )
                continue

            # get the global k ranges that this rank needs to send to current rank
            rank_remote_k_ranges_global = remote_rank_entry_per_rank_this_stage[
                rank
            ].remote_k_ranges_global
            k_ranges_global_send_to_rank = (
                host_k_ranges_global_this_rank.find_overlap_ranges(
                    rank_remote_k_ranges_global
                )
            )
            # make the global k ranges local w.r.t. self's send buffer
            k_ranges_local_send_to_rank = (
                host_k_ranges_global_this_rank.make_ranges_local(
                    k_ranges_global_send_to_rank
                )
            )
            # add to send transfer info for both global and local ones
            transfer_info_this_rank.k_ranges_global_send_to_each_rank_list.append(
                k_ranges_global_send_to_rank
            )
            transfer_info_this_rank.k_ranges_local_send_to_each_rank_list.append(
                k_ranges_local_send_to_rank
            )

        return transfer_info_this_rank

    def _init_transfer_info_per_rank(
        self,
        stage: int,
        transfer_info_this_rank: TransferInfo,
    ) -> list[TransferInfo]:
        # all gather initial recv/send transfer info for each rank
        transfer_info_per_rank: list[TransferInfo] = [None] * self.cp_size  # type: ignore
        dist.all_gather_object(
            transfer_info_per_rank, transfer_info_this_rank, group=self.cp_group_nccl
        )

        # sanity check:
        # for each rank pair (i≠j): (send_ranki, recv_rankj)
        #    whether the global k ranges that send_ranki needs to send to recv_rankj
        #    are equal to the ones that recv_rankj needs to recv from send_ranki
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
                ), (
                    f"The sanity check for transfer table at {stage=} failed:\n"
                    f"For rank pair ({send_rank=} {recv_rank=}), we got:\n"
                    f"{k_ranges_global_recv_from_send_rank=}\n"
                    f"{k_ranges_global_send_to_recv_rank=}"
                )

        return transfer_info_per_rank

    @nvtx.instrument_nvtx
    def calc_comm_meta(self) -> CommMeta:
        num_remote_tokens_list: list[int] = []
        group_cast_collective_args_list: list[GroupCastCollectiveArg] = []

        for transfer_table_this_stage, remote_rank_entry_per_rank_this_stage in zip(
            self.transfer_table_per_stage,
            self.remote_rank_entry_per_rank_per_stage,
        ):
            total_seqlen_host_k = remote_rank_entry_per_rank_this_stage[
                self.cp_rank
            ].host_k_ranges_global.seqlen

            num_remote_tokens = remote_rank_entry_per_rank_this_stage[
                self.cp_rank
            ].remote_k_ranges_global.seqlen

            group_cast_collective_arg = self._calc_group_cast_collective_arg(
                transfer_table_this_stage,
                total_seqlen_host_k,
            )

            num_remote_tokens_list.append(num_remote_tokens)
            group_cast_collective_args_list.append(group_cast_collective_arg)

        # build comm meta
        comm_meta = CommMeta(
            overlap_degree=self.overlap_degree,
            num_remote_tokens_per_overlap_stage=num_remote_tokens_list,
            group_cast_collective_args_list=group_cast_collective_args_list,
        )

        return comm_meta

    def _calc_group_cast_collective_arg(
        self,
        transfer_table: TransferTable,
        total_seqlen_host_k: int,
    ) -> GroupCastCollectiveArg:
        # retrieve group cast ranges for local k ranges that this rank needs to send to
        group_cast_ranges_local_send_to = GroupCastRanges(
            self.cp_size,
            [
                transfer_table.get_k_ranges_local_in_send_buf(
                    send_rank=self.cp_rank,
                    recv_rank=recv_rank,
                )
                for recv_rank in range(self.cp_size)
            ],
        )

        # split the local ranges into non-overlapped local ranges
        group_cast_ranges_local_send_to.split()

        # calc input split size list with dst indices list
        input_split_size_list: list[int] = []
        dst_indices_list: list[list[int]] = []

        last_end = 0
        for r in group_cast_ranges_local_send_to.ranges_group:
            r: AttnRangeWithRank  # type: ignore
            if r.start != last_end:  # [last_end, r.start) has no dest rank
                # FIXME: this branch is unreachable in the current test cases
                input_split_size_list.append(r.start - last_end)
                dst_indices_list.append([])

            input_split_size_list.append(r.size)
            dst_indices_list.append(list(r.rank_set))  # type: ignore
            last_end = r.end

        if last_end != total_seqlen_host_k:  # [last_end, seqlen) has no dest rank
            input_split_size_list.append(total_seqlen_host_k - last_end)
            dst_indices_list.append([])

        # retrieve group cast ranges for local k ranges that this rank needs to recv from
        group_cast_ranges_local_recv_from = GroupCastRanges(
            self.cp_size,
            [
                transfer_table.get_k_ranges_local_in_recv_buf(
                    send_rank=send_rank,
                    recv_rank=self.cp_rank,
                )
                for send_rank in range(self.cp_size)
            ],
        )

        # calc output split size list with src index list
        output_split_size_list = []
        src_index_list = []

        for r in group_cast_ranges_local_recv_from.ranges_group:
            r: AttnRangeWithRank  # type: ignore
            output_split_size_list.append(r.size)
            # NOTE: as for group cast semantics,
            # there's only one src rank that sends the corresponding data into
            # each non-overlapped range in recv buffer
            assert len(r.rank_set) == 1  # type: ignore
            src_index_list.append(r.rank_set.pop())  # type: ignore

        # build group cast collective arg
        group_cast_collective_arg = GroupCastCollectiveArg(
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
        )

        return group_cast_collective_arg

    @nvtx.instrument_nvtx
    def calc_attn_calc_meta(self) -> AttnCalcMeta:
        # check local attn calc
        assert all(
            slice is not None
            for slice in self.host_rank_entry_this_rank.attn_calc_host_slice_local_list
        )

        # check remote attn calc for each overlap stage
        for (
            remote_rank_entry_this_stage_this_rank
        ) in self.remote_rank_entry_per_stage_this_rank:
            assert all(
                slice is not None
                for slice in remote_rank_entry_this_stage_this_rank.attn_calc_remote_slice_local_list
            )

        # build attn calc meta
        attn_calc_meta = AttnCalcMeta(
            overlap_degree=self.overlap_degree,
            # build local attn args
            local_attn_arg=AttnArg(
                q_ranges=[
                    slice.q_range.to_naive_range()  # type: ignore
                    for slice in self.host_rank_entry_this_rank.attn_calc_host_slice_local_list
                ],
                k_ranges=[
                    slice.k_range.to_naive_range()  # type: ignore
                    for slice in self.host_rank_entry_this_rank.attn_calc_host_slice_local_list
                ],
                is_causal_mapping=[
                    slice.mask_type == AttnMaskType.CAUSAL
                    for slice in self.host_rank_entry_this_rank.attn_calc_host_slice_local_list
                ],
            ),
            # build remote attn args for each overlap stage
            remote_attn_args_list=[
                AttnArg(
                    q_ranges=[
                        slice.q_range.to_naive_range()  # type: ignore
                        for slice in remote_rank_entry_this_stage_this_rank.attn_calc_remote_slice_local_list
                    ],
                    k_ranges=[
                        slice.k_range.to_naive_range()  # type: ignore
                        for slice in remote_rank_entry_this_stage_this_rank.attn_calc_remote_slice_local_list
                    ],
                    is_causal_mapping=[
                        slice.mask_type == AttnMaskType.CAUSAL
                        for slice in remote_rank_entry_this_stage_this_rank.attn_calc_remote_slice_local_list
                    ],
                )
                for remote_rank_entry_this_stage_this_rank in self.remote_rank_entry_per_stage_this_rank
            ],
        )

        return attn_calc_meta

    def __repr__(self, title_len: int = 50) -> str:
        repr_contents = []

        repr_summary = self._repr_host_info(
            self.host_rank_entry_this_rank, title_len=title_len
        )
        repr_contents.append(repr_summary)

        for stage, (
            transfer_table_this_stage,
            remote_rank_entry_per_rank_this_stage,
        ) in enumerate(
            zip(
                self.transfer_table_per_stage,
                self.remote_rank_entry_per_rank_per_stage,
            )
        ):
            repr_this_stage = self._repr_remote_info_for_one_stage(
                stage,
                transfer_table_this_stage,
                remote_rank_entry_per_rank_this_stage,
                title_len=title_len,
            )

            repr_contents.append(repr_this_stage)

        return "\n\n".join(repr_contents)

    def _repr_host_info(
        self, host_rank_entry_this_rank: HostRankEntry, title_len: int = 50
    ) -> str:
        repr_info = []

        # add summary info title
        stage_title = "  Host Info  "
        repr_info.append("\n" + "=" * title_len + stage_title + "=" * title_len + "\n")

        host_q_ranges_global = host_rank_entry_this_rank.host_q_ranges_global
        host_k_ranges_global = host_rank_entry_this_rank.host_k_ranges_global
        remote_k_ranges_global = host_rank_entry_this_rank.remote_k_ranges_global
        attn_calc_slice_global_list = (
            host_rank_entry_this_rank.attn_calc_slice_global_list
        )
        attn_calc_host_slice_local_list = (
            host_rank_entry_this_rank.attn_calc_host_slice_local_list
        )

        repr_info.append(f"host_q_ranges_global: {host_q_ranges_global}")
        repr_info.append(f"host_k_ranges_global: {host_k_ranges_global}")
        repr_info.append(f"remote_k_ranges_global: {remote_k_ranges_global}")
        repr_info.append(f"attn_calc_slice_global_list: {attn_calc_slice_global_list}")
        repr_info.append(
            f"attn_calc_host_slice_local_list: {attn_calc_host_slice_local_list}"
        )

        return "\n".join(repr_info)

    def _repr_remote_info_for_one_stage(
        self,
        stage: int,
        transfer_table_this_stage: TransferTable,
        remote_rank_entry_per_rank_this_stage: list[RemoteRankEntry],
        title_len: int = 50,
    ) -> str:
        # 计算每个单元格需要的最大宽度
        cell_widths = [[0] * self.cp_size for _ in range(self.cp_size)]
        for send_rank in range(self.cp_size):
            for recv_rank in range(self.cp_size):
                send_str = f"send: {transfer_table_this_stage.get_k_ranges_local_in_send_buf(send_rank, recv_rank)}"
                recv_str = f"recv: {transfer_table_this_stage.get_k_ranges_local_in_recv_buf(send_rank, recv_rank)}"
                global_str = f"global: {transfer_table_this_stage.get_k_ranges_global(send_rank, recv_rank)}"

                width = max(len(send_str), len(recv_str), len(global_str))
                cell_widths[send_rank][recv_rank] = width

        # 计算每列的最大宽度
        col_widths = [
            max(
                max(cell_widths[row][col] for row in range(self.cp_size)),
                len(
                    "host_k_ranges_global: "
                    f"{remote_rank_entry_per_rank_this_stage[col].host_k_ranges_global}"
                ),
                len(
                    "remote_k_ranges_global: "
                    f"{remote_rank_entry_per_rank_this_stage[col].remote_k_ranges_global}"
                ),
                len(
                    "attn_calc_remote_slice_local_list: "
                    f"{remote_rank_entry_per_rank_this_stage[col].attn_calc_remote_slice_local_list}"
                ),
            )
            for col in range(self.cp_size)
        ]

        # 计算表格的总宽度（考虑到每列分隔符 " | " 以及每行的"row xx |"前缀）
        table_width = (
            sum(col_widths) + 4 * (self.cp_size - 1) + 7
        )  # 每列间隔宽度4 + row xx | 前缀宽度为7

        # 构建表格
        repr_info_this_stage = []

        # 添加overlap stage title分割线
        stage_title = f"  Remote Info for Stage {stage}  "
        repr_info_this_stage.append(
            "\n" + "=" * title_len + stage_title + "=" * title_len + "\n"
        )

        # 添加列标题行（扩展为5行高度）
        repr_info_this_stage.append("\n" + "-" * table_width)

        # 第一行：列号
        header_cells = [f"col{j:2d}".center(col_widths[j]) for j in range(self.cp_size)]
        repr_info_this_stage.append("r/c   | " + " | ".join(header_cells) + " |")

        # 第二行：host_k_ranges_global
        host_cells = [
            f"host_k_ranges_global: {remote_rank_entry_per_rank_this_stage[j].host_k_ranges_global}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        repr_info_this_stage.append("      | " + " | ".join(host_cells) + " |")

        # 第三行：remote_k_ranges_global
        remote_cells = [
            f"remote_k_ranges_global: {remote_rank_entry_per_rank_this_stage[j].remote_k_ranges_global}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        repr_info_this_stage.append("      | " + " | ".join(remote_cells) + " |")

        # 第四行：attn_calc_remote_slice_local_list
        remote_slice_cells = [
            "attn_calc_remote_slice_local_list: "
            f"{remote_rank_entry_per_rank_this_stage[j].attn_calc_remote_slice_local_list}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        repr_info_this_stage.append("      | " + " | ".join(remote_slice_cells) + " |")

        # 添加分割线
        repr_info_this_stage.append("-" * table_width)

        # 添加每一行
        for send_rank in range(self.cp_size):
            # 处理每个单元格的三行内容
            cell_lines = []
            for recv_rank in range(self.cp_size):
                col_width = col_widths[recv_rank]
                cell_content = [
                    f"send: {transfer_table_this_stage.get_k_ranges_local_in_send_buf(send_rank, recv_rank)}".ljust(
                        col_width
                    ),
                    f"recv: {transfer_table_this_stage.get_k_ranges_local_in_recv_buf(send_rank, recv_rank)}".ljust(
                        col_width
                    ),
                    f"global: {transfer_table_this_stage.get_k_ranges_global(send_rank, recv_rank)}".ljust(
                        col_width
                    ),
                ]
                cell_lines.append(cell_content)

            # 组装每行的三行内容
            for line_idx in range(3):
                prefix = f"row{send_rank:2d} |" if line_idx == 0 else "      |"
                line = [cell_lines[j][line_idx] for j in range(self.cp_size)]
                repr_info_this_stage.append(f"{prefix} " + " | ".join(line) + " |")

            repr_info_this_stage.append("-" * table_width)  # 每行后面添加分割线

        return "\n".join(repr_info_this_stage)
