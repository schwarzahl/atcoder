package abc084;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveD();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		System.out.println(48 - N);
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		int A = sc.nextInt();
		int B = sc.nextInt();
		String C = sc.next();
		for (int i = 0; i < A; i++) {
			if (C.charAt(i) == '-') {
				System.out.println("No");
				return;
			}
		}
		if (C.charAt(A) != '-') {
			System.out.println("No");
			return;
		}
		for (int i = A + 1; i < A + B + 1; i++) {
			if (C.charAt(i) == '-') {
				System.out.println("No");
				return;
			}
		}
		System.out.println("Yes");
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long[] C = new long[N + 1];
		long[] S = new long[N + 1];
		long[] F = new long[N + 1];
		for (int i = 1; i < N; i++) {
			C[i] = sc.nextLong();
			S[i] = sc.nextLong();
			F[i] = sc.nextLong();
		}
		for (int start = 1; start <= N; start++) {
			long current_time = 0L;
			for (int i = start; i < N; i++) {
				if (current_time < S[i]) {
					current_time = S[i];
				}
				if (current_time % F[i] > 0) {
					current_time = (current_time + F[i]) / F[i] * F[i];
				}
				current_time += C[i];
			}
			System.out.println(current_time);
		}
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int Q = sc.nextInt();
		PrimeNumberUtils pnu = new PrimeNumberUtils(100001);
		boolean[] n2017 = new boolean[100001];
		int[] sum = new int[100001];
		for (int i = 1; i < 100001; i += 2) {
			n2017[i] = pnu.isPrime(i) && pnu.isPrime((i + 1) / 2);
		}
		sum[0] = 0;
		for (int i = 1; i < 100001; i++) {
			sum[i] = sum[i - 1] + (n2017[i] ? 1 : 0);
		}
		for (int i = 0; i < Q; i++) {
			int l = sc.nextInt();
			int r = sc.nextInt();
			System.out.println(sum[r] - sum[l - 1]);
		}
	}

	/**
	 * 素数のユーティリティ
	 */
	class PrimeNumberUtils {
		boolean[] isPrimeArray;
		List<Long> primes;

		/**
		 * 素数判定の上限となる値を指定してユーティリティを初期化
		 * @param limit 素数判定の上限
		 */
		public PrimeNumberUtils(long limit) {
			primes = new ArrayList<>();
			isPrimeArray = new boolean[limit < Integer.MAX_VALUE ? (int)limit : Integer.MAX_VALUE];
			if (limit > 2) {
				primes.add(2L);
				isPrimeArray[2] = true;
			}

			for (long i = 3; i < limit; i += 2L) {
				if (isPrime(i, primes)) {
					primes.add(i);
					if (i < Integer.MAX_VALUE) {
						isPrimeArray[(int) i] = true;
					}
				}
			}
		}

		public List<Long> getPrimeNumberList() {
			return primes;
		}

		public boolean isPrime(int n) {
			return isPrimeArray[n];
		}

		public boolean isPrime(long n) {
			return primes.contains(n);
		}

		private boolean isPrime(long n, List<Long> primes) {
			for (long prime : primes) {
				if (n % prime == 0) {
					return false;
				}
				if (prime > Math.sqrt(n)) {
					break;
				}
			}
			return true;
		}
	}

	interface Graph {
		void link(int from, int to, long cost);
		Optional<Long> getCost(int from, int to);
		int getVertexNum();
	}

	interface FlowResolver {
		long maxFlow(int from, int to);
	}

	/**
	 * グラフの行列による実装
	 * 接点数の大きいグラフで使うとMLEで死にそう
	 */
	class ArrayGraph implements Graph {
		private Long[][] costArray;
		private int vertexNum;

		public ArrayGraph(int n) {
			costArray = new Long[n][];
			for (int i = 0; i < n; i++) {
				costArray[i] = new Long[n];
			}
			vertexNum = n;
		}

		@Override
		public void link(int from, int to, long cost) {
			costArray[from][to] = new Long(cost);
		}

		@Override
		public Optional<Long> getCost(int from, int to) {
			return Optional.ofNullable(costArray[from][to]);
		}

		@Override
		public int getVertexNum() {
			return vertexNum;
		}
	}

	/**
	 * DFS(深さ優先探索)による実装
	 * 計算量はO(E*MaxFlow)のはず (E:辺の数, MaxFlow:最大フロー)
	 */
	class DfsFlowResolver implements FlowResolver {
		private Graph graph;
		public DfsFlowResolver(Graph graph) {
			this.graph = graph;
		}

		/**
		 * 最大フロー(最小カット)を求める
		 * @param from 始点(source)のID
		 * @param to 終点(target)のID
		 * @return 最大フロー(最小カット)
		 */
		public long maxFlow(int from, int to) {
			long sum = 0L;
			long currentFlow;
			do {
				currentFlow = flow(from, to, Long.MAX_VALUE / 3, new boolean[graph.getVertexNum()]);
				sum += currentFlow;
			} while (currentFlow > 0);
			return sum;
		}

		/**
		 * フローの実行 グラフの更新も行う
		 * @param from 現在いる節点のID
		 * @param to 終点(target)のID
		 * @param current_flow ここまでの流量
		 * @param passed 既に通った節点か否かを格納した配列
		 * @return 終点(target)に流した流量/戻りのグラフの流量
		 */
		private long flow(int from, int to, long current_flow, boolean[] passed) {
			passed[from] = true;
			if (from == to) {
				return current_flow;
			}
			for (int id = 0; id < graph.getVertexNum(); id++) {
				if (passed[id]) {
					continue;
				}
				Optional<Long> cost = graph.getCost(from, id);
				if (cost.orElse(0L) > 0) {
					long nextFlow = current_flow < cost.get() ? current_flow : cost.get();
					long returnFlow = flow(id, to, nextFlow, passed);
					if (returnFlow > 0) {
						graph.link(from, id, cost.get() - returnFlow);
						graph.link(id, from, graph.getCost(id, from).orElse(0L) + returnFlow);
						return returnFlow;
					}
				}
			}
			return 0L;
		}
	}

	/**
	 * 1-indexedのBIT配列
	 */
	class BinaryIndexedTree {
		private long[] array;

		public BinaryIndexedTree(int size) {
			this.array = new long[size + 1];
		}

		/**
		 * 指定した要素に値を加算する
		 * 計算量はO(logN)
		 * @param index 加算する要素の添字
		 * @param value 加算する量
		 */
		public void add(int index, long value) {
			for (int i = index; i < array.length; i += (i & -i)) {
				array[i] += value;
			}
		}

		/**
		 * 1〜指定した要素までの和を取得する
		 * 計算量はO(logN)
		 * @param index 和の終端
		 * @return 1〜indexまでの和
		 */
		public long getSum(int index) {
			long sum = 0L;
			for (int i = index; i > 0; i -= (i & -i)) {
				sum += array[i];
			}
			return sum;
		}
	}

	interface UnionFind {
		void union(int A, int B);
		boolean judge(int A, int B);
	}

	/**
	 * 配列によるUnionFindの実装
	 */
	class ArrayUnionFind implements UnionFind {
		int[] parent;
		int[] rank;
		public ArrayUnionFind(int size) {
			parent = new int[size];
			for (int i = 0; i < size; i++) {
				parent[i] = i;
			}
			rank = new int[size];
		}

		@Override
		public void union(int A, int B) {
			int rootA = root(A);
			int rootB = root(B);
			if (rootA != rootB) {
				if (rank[rootA] < rank[rootB]) {
					parent[rootA] = rootB;
				} else {
					parent[rootB] = rootA;
					if (rank[rootA] == rank[rootB]) {
						rank[rootA]++;
					}
				}
			}
		}

		@Override
		public boolean judge(int A, int B) {
			return root(A) == root(B);
		}

		protected int root(int id) {
			if (parent[id] == id) {
				return id;
			}
			parent[id] = root(parent[id]);
			return parent[id];
		}
	}
}