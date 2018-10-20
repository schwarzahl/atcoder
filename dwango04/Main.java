package dwango04;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.IntStream;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveC();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		int s = sc.nextInt();
		if (s / 1000 == (s / 10) % 10 && (s / 100) % 10 == s % 10) {
			System.out.println("Yes");
		} else {
			System.out.println("No");
		}
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		String s = sc.next();
		int count = 0;
		int max = 0;
		for (char c : s.toCharArray()) {
			if (c == '2') {
				count++;
				if (max < count) {
					max = count;
				}
			} else {
				count--;
				if (count < 0) {
					break;
				}
			}
		}
		if (count == 0) {
			System.out.println(max);
		} else {
			System.out.println(-1);
		}
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();
		int[] killA = new int[N + 1];
		int[] killB = new int[M + 1];
		int killAsum = 0;
		killA[0] = -1;
		for (int i = 1; i <= N; i++) {
			killA[i] = sc.nextInt();
			killAsum += killA[i];
		}
		int killBsum = 0;
		killB[0] = -1;
		for (int i = 1; i <= M; i++) {
			killB[i] = sc.nextInt();
			killBsum += killB[i];
		}
		long sumA = calc(1, 0, killBsum, killA);
		long sumB = calc(1, 0, killAsum, killB);
		System.out.println((sumA * sumB) % 1000000007L);
	}

	private long calc(int current_id, int death, int restDeath, int[] kills) {
		/*
		if (current_id >= kills.length) {
			if (restDeath == 0) {
				return 1L;
			} else {
				return 0L;
			}
		}
		*/
		if (current_id == kills.length - 1) {
			if (kills[current_id] == kills[current_id - 1] && restDeath + 1 - death <= 0) {
				return 0L;
			}
			return 1L;
			/*
			if (kills[current_id] == kills[current_id - 1]) {
				return Math.max(restDeath + 1 - death, 0);
			}
			return restDeath + 1;
			*/
		}
		long sum = 0L;
		for (int nextDeath = 0; nextDeath <= restDeath; nextDeath++) {
			if (kills[current_id] == kills[current_id - 1] && death > nextDeath) {
				continue;
			}
			sum = (sum + calc(current_id + 1, nextDeath, restDeath - nextDeath, kills)) % 1000000007L;
		}
		return sum;
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long[] x = new long[N + 1];
		for (int i = 1; i <= N; i++) {
			x[i] = sc.nextLong();
		}
		long[] sum = new long[N + 1];
		int[] parent = new int[N + 1];
		parent[1] = 0;
		sum[1] += x[1];
		for (int i = 2; i <= N; i++) {
			parent[i] = sc.nextInt();
			sum[i] += x[i];
			sum[parent[i]] += x[i];
		}
		long max = 0L;
		for (long tmp : sum) {
			if (max < tmp) {
				max = tmp;
			}
		}
		System.out.println(max);
	}

	private void solveE() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int Q = sc.nextInt();
		boolean[][] graph = new boolean[N + 1][];
		for (int i = 0; i <= N; i++) {
			graph[i] = new boolean[N + 1];
		}
		for (int i = 1; i < N; i++) {
			int a = sc.nextInt();
			int b = sc.nextInt();
			graph[a][b] = true;
			graph[b][a] = true;
		}
		boolean[] exist = new boolean[N + 1];
		Arrays.fill(exist, true);
		for (int tryNum = 0; tryNum < Q; tryNum++){
			int max1 = 0;
			int max1_id = 1;
			for (int i = 1; i <= N; i++) {
				if (exist[i]) {
					max1_id = i;
					break;
				}
			}
			{
				int[] dist = new int[N + 1];
				Arrays.fill(dist, N + 1);
				search(1, 0, dist, graph, N);
				for (int i = 2; i <= N; i++) {
					if (exist[i] && max1 < dist[i]) {
						max1 = dist[i];
						max1_id = i;
					}
				}
			}
			int max2 = 0;
			int max2_id = max1_id;
			int[] dist1 = new int[N + 1];
			Arrays.fill(dist1, N + 1);
			search(max1_id, 0, dist1, graph, N);
			for (int i = 2; i <= N; i++) {
				if (exist[i] && max2 < dist1[i]) {
					max2 = dist1[i];
					max2_id = i;
				}
			}
			int[] dist2 = new int[N + 1];
			Arrays.fill(dist2, N + 1);
			search(max2_id, 0, dist2, graph, N);
			System.out.println("? " + max1_id + " " + max2_id);
			int near = sc.nextInt();
			if (near == 0) {
				for (int i = 1; i <= N; i++) {
					if (dist1[i] != dist2[i]) {
						exist[i] = false;
					}
				}
			} else if (near == max1_id) {
				for (int i = 1; i <= N; i++) {
					if (dist1[i] >= dist2[i]) {
						exist[i] = false;
					}
				}
			} else if (near == max2_id) {
				for (int i = 1; i <= N; i++) {
					if (dist1[i] <= dist2[i]) {
						exist[i] = false;
					}
				}
			}
			int answer = 0;
			int answer_num = 0;
			for (int i = 1; i <= N; i++) {
				if (exist[i]) {
					answer = i;
					answer_num++;
				}
			}
			if (answer_num == 1) {
				System.out.println("! " + answer);
				return;
			}
		}
	}

	private void search(int current, int level, int[] dist, boolean[][] graph, int N) {
		if (dist[current] < level) {
			return;
		}
		dist[current] = level;
		for (int i = 1; i <= N; i++) {
			if (graph[current][i]) {
				search(i, level + 1, dist, graph, N);
			}
		}
	}

	private void solveF() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		System.out.println(N);
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
		Set<Integer> getSet(int id);
	}

	/**
	 * ArrayUnionFindの拡張
	 * MapSetで根の添字から根にぶら下がる頂点の集合が取得できるようにした
	 * getSetメソッドをO(logN * logN)に落とせているはず
	 * ただしunionメソッドは2倍の計算量になっているので注意(オーダーは変わらないはず)
	 */
	class SetUnionFind extends ArrayUnionFind {
		Map<Integer, Set<Integer>> map;
		public SetUnionFind(int size) {
			super(size);
			map = new HashMap<>();
			for (int i = 0; i < size; i++) {
				map.put(i, new HashSet<>());
				map.get(i).add(i);
			}
		}

		@Override
		protected void unionTo(int source, int dest) {
			super.unionTo(source, dest);
			map.get(dest).addAll(map.get(source));
		}

		@Override
		public Set<Integer> getSet(int id) {
			return map.get(root(id));
		}
	}

	/**
	 * 配列によるUnionFindの実装
	 * getSetメソッドはO(NlogN)なのでTLEに注意
	 */
	class ArrayUnionFind implements UnionFind {
		int[] parent;
		int[] rank;
		int size;
		public ArrayUnionFind(int size) {
			parent = new int[size];
			for (int i = 0; i < size; i++) {
				parent[i] = i;
			}
			rank = new int[size];
			this.size = size;
		}

		@Override
		public void union(int A, int B) {
			int rootA = root(A);
			int rootB = root(B);
			if (rootA != rootB) {
				if (rank[rootA] < rank[rootB]) {
					unionTo(rootA, rootB);
				} else {
					unionTo(rootB, rootA);
					if (rank[rootA] == rank[rootB]) {
						rank[rootA]++;
					}
				}
			}
		}

		protected void unionTo(int source, int dest) {
			parent[source] = dest;
		}

		@Override
		public boolean judge(int A, int B) {
			return root(A) == root(B);
		}

		@Override
		public Set<Integer> getSet(int id) {
			Set<Integer> set = new HashSet<>();
			IntStream.range(0, size).filter(i -> judge(i, id)).forEach(set::add);
			return set;
		}

		protected int root(int id) {
			if (parent[id] == id) {
				return id;
			}
			parent[id] = root(parent[id]);
			return parent[id];
		}
	}

	/**
	 * 素数のユーティリティ
	 */
	class PrimeNumberUtils {
		boolean[] isPrimeArray;
		List<Integer> primes;

		/**
		 * 素数判定の上限となる値を指定してユーティリティを初期化
		 * @param limit 素数判定の上限(この値以上が素数であるか判定しない)
		 */
		public PrimeNumberUtils(int limit) {
			if (limit > 10000000) {
				System.err.println("上限の値が高すぎるため素数ユーティリティの初期化でTLEする可能性が大変高いです");
			}
			primes = new ArrayList<>();
			isPrimeArray = new boolean[limit];
			if (limit > 2) {
				primes.add(2);
				isPrimeArray[2] = true;
			}

			for (int i = 3; i < limit; i += 2) {
				if (isPrime(i, primes)) {
					primes.add(i);
					isPrimeArray[i] = true;
				}
			}
		}

		public List<Integer> getPrimeNumberList() {
			return primes;
		}

		public boolean isPrime(int n) {
			return isPrimeArray[n];
		}

		private boolean isPrime(int n, List<Integer> primes) {
			for (int prime : primes) {
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
}