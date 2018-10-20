package arc086;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveE();
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int K = sc.nextInt();
		int A[] = new int[200001];

		for (int i = 0; i < N; i++) {
			A[sc.nextInt()]++;
		}
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i <= 200000; i++) {
			if (A[i] > 0) {
				list.add(A[i]);
			}
		}
		Collections.sort(list);
		int erase = list.size() - K;
		if (erase > 0) {
			System.out.println(list.subList(0, erase).stream().mapToInt(a -> a).sum());
		} else {
			System.out.println(0);
		}
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int a[] = new int[N+1];
		int max = 0;
		int max_id = -1;
		int min = 0;
		int min_id = -1;
		for (int i = 1; i <= N; i++) {
			a[i] = sc.nextInt();
			if (max < a[i]) {
				max = a[i];
				max_id = i;
			}
			if (min > a[i]) {
				min = a[i];
				min_id = i;
			}
		}
		if (max >= -min && max_id != -1) {
			System.out.println(2 * N - 2);
			for (int i = 1; i <= N; i++) {
				if (i == max_id) {
					continue;
				}
				System.out.println(max_id + " " + i);
			}
			for (int i = 1; i <= N - 1; i++) {
				System.out.println(i + " " + (i+1));
			}
		} else if (max <= -min && min_id != -1) {
			System.out.println(2 * N - 2);
			for (int i = 1; i <= N; i++) {
				if (i == min_id) {
					continue;
				}
				System.out.println(min_id + " " + i);
			}
			for (int i = N; i > 1; i--) {
				System.out.println(i + " " + (i-1));
			}
		} else {
			System.out.println(0);
		}
	}

	private void solveE() {
		final long MOD_NUM = 1000000007L;
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int[] parent = new int[N + 1];
		Map<Integer, List<Integer>> child = new HashMap<>();
		int[] level = new int[N + 1];
		int[] level_num = new int [N + 1];
		level_num[0] = 1;
		for (int i = 1; i <= N; i++) {
			parent[i] = sc.nextInt();
			level[i] = level[parent[i]] + 1;
			level_num[level[i]]++;
			if (!child.containsKey(parent[i])) {
				child.put(parent[i], new ArrayList<>());
			}
			child.get(parent[i]).add(i);
		}
		long ans = 0L;
		for (int d = 0; d <= N; d++) {
			if (level_num[d] == 0) {
				continue;
			}
			long[][] dp = new long[N + 1][];
			for (int i = 0; i <= N; i++) {
				dp[i] = new long[2];
			}
			for (int i = N; i >= 0; i--) {
				if (level[i] == d) {
					dp[i][0] = 1;
					dp[i][1] = 1;
				} else {
					long sum = 0L;
					long all_mul = 1L;
					if (child.containsKey(i)) {
						for (int child_id : child.get(i)) {
							long mul = 1L;
							for (int child_id2 : child.get(i)) {
								mul = (mul * dp[child_id][child_id == child_id2 ? 1 : 0]) % MOD_NUM;
							}
							sum += mul;
							all_mul = (((dp[child_id][0] + dp[child_id][1]) % MOD_NUM) * all_mul) % MOD_NUM;
						}
					}
					if (sum > 0) {
						dp[i][0] = (all_mul - sum + MOD_NUM) % MOD_NUM;
						dp[i][1] = sum % MOD_NUM;
					}
				}
			}
			ans += (dp[0][1] * pow2(N + 1 - level_num[d], MOD_NUM)) % MOD_NUM;
			System.err.println(d + ":" + dp[0][1]);
		}
		System.out.println(ans);
	}

	private long pow2(int n, long MOD_NUM) {
		long ret = 1L;
		for (int i = 0; i < n; i++) {
			ret = (ret * 2) % MOD_NUM;
		}
		return ret;
	}
}