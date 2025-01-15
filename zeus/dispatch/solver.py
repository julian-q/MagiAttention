# mypy: ignore-errors
import heapq
from enum import Enum
from typing import List, Tuple

import numpy as np
import torch.nn as nn


class DispatchAlgorithm(Enum):
    """This enum is used to specify the algorithm for balanced dispatching"""

    LOWER_BOUND = "lower_bound"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    BINARY_SEARCH = "binary_search"
    MIN_HEAP = "min_heap"


class DispatchSolver(nn.Module):
    """The implementation of the algorithms for balanced dispatching specified by `alg`."""

    def __init__(
        self,
        alg: DispatchAlgorithm = DispatchAlgorithm.MIN_HEAP,
    ) -> None:
        super().__init__()

        self.alg = alg

        self.solve_func = {
            DispatchAlgorithm.LOWER_BOUND: self._solve_with_lb,
            DispatchAlgorithm.DYNAMIC_PROGRAMMING: self._solve_with_dp,
            DispatchAlgorithm.BINARY_SEARCH: self._solve_with_bs,
            DispatchAlgorithm.MIN_HEAP: self._solve_with_minhp,
        }[self.alg]

        # return values
        self.max_workload = None
        self.workloads = None
        self.partitions = None

    def solve(
        self,
        jobs: List[float],
        k: int,
        **kwargs,
    ) -> Tuple[float, List[float] | None, List[List[int]] | None]:
        """Dispatch jobs to k buckets, to make the workload of each bucket as balanced as possible

        Args:
            jobs (List[float]): the workload of each job
            k (int): the number of buckets
            kwargs (dict | None): additional arguments for specific algorithms
        Returns:
            max_workload (float): the maximum workload of any bucket
            workloads (List[float], optional): the workload of each bucket
            partitions (List[List[int]], optional): the partition of jobs, where each element is a list of job indices,
                among them any two elements are mutually exclusive
        """

        self.solve_func(jobs, k, **kwargs)

        return self.max_workload, self.workloads, self.partitions

    def _bs_backtrack(self, jobs: List[float], idx: int, limit: int) -> bool:
        """Assign the current job with index `idx` to some bucket whose workload does not exceed `limit`
        and then recursively assign the remaining jobs,
        until all jobs have been assigned or no solution is found

        Args:
            jobs (List[float]): the workload of each job
            idx (int): the index of the current job
            limit (int): the maximum workload of each bucket
        Returns:
            bool: whether a solution is found
        """

        if idx >= len(jobs):
            return True  # all jobs have been assigned

        current_job_workload = jobs[idx]
        for partition in self.partitions:
            partition_sum = sum(jobs[idx] for idx in partition)

            if partition_sum + current_job_workload <= limit:
                partition.append(idx)
                if self._bs_backtrack(jobs, idx + 1, limit):
                    return True
                partition.pop()

            # if the current bucket has not been assigned any work,
            # or assigning the current work exactly reaches the limit,
            # there is no need to try more buckets
            if partition_sum == 0 or partition_sum + current_job_workload == limit:
                break

        return False

    def _bs_check(self, jobs: List[float], k: int, limit: int) -> bool:
        """Check if it is possible to assign all jobs to k buckets,
        such that the workload of each bucket does not exceed `limit`

        Args:
            jobs (List[float]): the workload of each job
            k (int): the number of buckets
            limit (int): the maximum workload of each bucket
        Returns:
            bool: whether a solution is found
        """
        self.partitions = [[] for _ in range(k)]
        return self._bs_backtrack(jobs, 0, limit)

    def _solve_with_lb(self, jobs: List[float], k: int, **kwargs) -> None:
        """Lower bound for the job partition to minimize the maximum workload
        NOTE: this algorithm is just to get the lower bound of the answer, which might be invalid

        Args:
            jobs (List[float]): the workload of each job
            k (int): the number of buckets, by which the number of jobs should be divisible
            kwargs (dict | None): additional arguments
        """
        self.max_workload = sum(jobs) / k

    def _solve_with_bs(self, jobs: List[float], k: int, **kwargs) -> None:
        """Binary search for the job partition to minimize the maximum workload
        NOTE: this algorithm has no constraint on the number of jobs dispatched in each bucket

        Args:
            jobs (List[float]): the workload of each job
            k (int): the number of buckets
            kwargs (dict | None): additional arguments

        Time complexity:
            - O(nlogn + log(S−M) x k**n), where:
                - n is the length of the jobs array
                - S is the sum of all elements in the jobs array
                - M is the maximum element in the jobs array
                - log(S−M) is the number of possible partitions
                - k**n is the number of possible combinations of partitions
            - NOTE: this is the worst-case time complexity,
                but in practice it is often much faster due to search pruning

        Space complexity:
            - O(n), where:
                - n is the length of the jobs array
                - and the space is mainly used by the recursive stack
        """

        # sort jobs in descending order
        sorted_indices = np.argsort(jobs)[::-1]
        jobs = [jobs[i] for i in sorted_indices]

        # init the range of binary search
        left = jobs[0]  # left: the maximum size of a single job
        right = sum(jobs)  # right: the sum of all jobs

        while left < right:
            mid = (left + right) // 2
            if self._bs_check(jobs, k, mid):  # check if mid is a valid solution
                right = mid  # if yes, reduce the range to the right half
            else:
                left = mid + 1  # if no, increase the range to the left half

        self.max_workload = left
        self.workloads = [sum(jobs[i] for i in p) for p in self.partitions]
        self.partitions = [[sorted_indices[i] for i in p] for p in self.partitions]

    def _solve_with_dp(self, jobs: List[float], k: int, **kwargs) -> None:
        """Dynamic programming for the job partition to minimize the maximum workload
        NOTE: this algorithm has no constraint on the number of jobs dispatched in each bucket,
        and it only returns the scalar answer of the maximum workload, w/o neither the partition or the workloads

        Args:
            jobs (List[float]): the workload of each job
            k (int): the number of buckets
            kwargs (dict | None): additional arguments

        Time complexity:
            - O(k x 3**n), where:
                - n is the length of the jobs array
                - k is the number of buckets
                - firstly of all, we need O(2**n) time to pre-process the sum array / init the dp array
                - then, we need to go through a double for-loop to do dp transition,
                    - the outer for-loop has O(k) iterations
                    - the inner for-loop has O(2**n) states, each state will tranverse all of its substates,
                    - so there will be ∑(f:0->n) C(n,f) * 2**f = (1+2)**n = 3**n iterations,
                        where f is the number of 1s in the state
                - therefore, the total iterations in the transition is O(k x 3**n)
                - finally, the total time complexity is O(k x 3**n + 2**n) = O(k x 3**n)

        Space complexity:
            - O(2**n), where:
                - n is the length of the jobs array
                - firstly of all, we need a sum array, which costs O(2**n)
                - and a dp array, which costs O(2**n) if we do the state compression
        """

        # init dp array, where:
        # dp(i)[j]: the maximum workload if we assign the jobs to the first i buckets
        # when the state (the assigned jobs' bit-map) is j
        m = 1 << len(jobs)  # m = 2**n
        dp = [0] * m

        # init: dp(0)[j] = sum[j] = sum(jobs in j)
        for i, v in enumerate(jobs):
            bit = 1 << i
            for j in range(bit):
                dp[bit | j] = dp[j] + v
        sum = dp.copy()

        # dp transition, where:
        # dp(i)[j] = min(max(dp(i-1)[j ^ s], sum(jobs in s))) for each s,
        # where s is the substate of j, (j ^ s) is the rest state
        # which indicates that we assign the jobs in s to the i-th bucket,
        # thus assign the jobs in (j ^ s) to the first (i-1)-th bucket
        for i in range(1, k):
            for j in range(m - 1, 0, -1):
                s = j
                while s:  # for each substate s of j until s = 0
                    dp[j] = min(dp[j], max(dp[j ^ s], sum[s]))
                    s = (s - 1) & j  # transfer to the next substate

        # dp(k-1)[m-1] is the answer, since we already assign all jobs to k buckets
        self.max_workload = dp[-1]

        # NOTE: since there's no traceback, we can't recover the partition, nor the workload of each bucket
        self.workloads = None
        self.partitions = None

    def _solve_with_minhp(self, jobs: List[float], k: int, **kwargs) -> None:
        """Greedy algorithm using a min-heap for the job partition to minimize the maximum workload,
        with the hard constraint that the number of jobs dispatched in each bucket should be the same
        NOTE: this algorithm is not guaranteed to find the optimal solution

        Args:
            jobs (List[float]): the workload of each job
            k (int): the number of buckets, by which the number of jobs should be divisible
            kwargs (dict | None): additional arguments

        Time complexity:
            - O(nlogn + nklogk + k), where:
                - n is the length of the jobs array
                - k is the number of buckets
                - firstly of all, we need O(nlogn) time to sort the jobs array
                - then, we need O(k) time to init the min-heap
                - then, we need O(nklogk) time to do the greedy assignment for each job
                - where each assignment at most costs O(klogk) times of heap-push and O(klogk) times of heap-pop

        Space complexity:
            - O(n + k), where:
                - n is the length of the jobs array
                - k is the number of buckets
                - the job array needs O(n) space
                - the min-heap and other auxiliary data needs O(k) space
        """

        # check the job number constraint
        n = len(jobs)
        assert (
            n % k == 0
        ), f"The number of jobs ({n}) should be divisible by k ({k}) for this algorithm."
        bucket_num_limit = n // k

        # sort jobs in descending order
        sorted_indices = np.argsort(jobs)[::-1]
        jobs = [jobs[i] for i in sorted_indices]

        # init the job partition and its workloads
        bucket_nums = [0] * k
        self.workloads = [0] * k
        self.partitions = [[] for _ in range(k)]

        # init the min-heap with size of k
        # where each heap element is a tuple: (current_sum, bucket_index)
        heap = []
        for i in range(k):
            heap.append((0, i))
        heapq.heapify(heap)

        # assign each job to a bucket
        for job_idx, job_workload in enumerate(jobs):
            while (
                heap
            ):  # find the bucket with the smallest sum that is not full to assign this job
                current_sum, bucket_idx = heapq.heappop(heap)
                if bucket_nums[bucket_idx] < bucket_num_limit:
                    # assign the job to this bucket
                    self.partitions[bucket_idx].append(job_idx)
                    self.workloads[bucket_idx] = current_sum + job_workload
                    bucket_nums[bucket_idx] += 1
                    # push the updated sum of this bucket back into the heap
                    heapq.heappush(heap, (self.workloads[bucket_idx], bucket_idx))
                    break
            else:  # if no bucket is found for this job (which shouldn't happen), raise an error
                raise RuntimeError(f"No bucket is found for the job {job_idx}.")

        self.max_workload = max(self.workloads)
        self.partitions = [[sorted_indices[i] for i in p] for p in self.partitions]
