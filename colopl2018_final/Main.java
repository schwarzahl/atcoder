package atcoder.colopl2018_final;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Scanner;
import java.util.Set;
import java.util.Stack;
import java.util.stream.IntStream;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveC();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		long N = sc.nextLong();
		String S = sc.next();
		List<Long> list = new ArrayList<>();
		long a_num = 0;
		for (char c : S.toCharArray()) {
			if (c == 'A') {
				a_num++;
			}
			if (c == 'B') {
				list.add(a_num);
				a_num = 0L;
			}
		}
		list.add(a_num);
		if (list.size() > 1) {
			long ans = 0L;
			if (N > 2) {
				for (int j = 1; j < list.size() - 1; j++) {
					long tmp = list.get(j);
					ans += tmp * (tmp + 1L) / 2L;
				}
				long tmp_2 = (list.get(0) + list.get(list.size() - 1));
				ans += tmp_2 * (tmp_2 + 1L) / 2L;
				ans *= N - 2L;
			}
			if (N > 1) {
				for (int i = 0; i < list.size() - 1; i++) {
					long tmp = list.get(i);
					ans += tmp * (tmp + 1L) / 2L;
				}
				long tmp_2 = (list.get(0) + list.get(list.size() - 1));
				ans += tmp_2 * (tmp_2 + 1L) / 2L;
				for (int i = 1; i < list.size(); i++) {
					long tmp = list.get(i);
					ans += tmp * (tmp + 1L) / 2L;
				}
			} else {
				for (int i = 0; i < list.size(); i++) {
					long tmp = list.get(i);
					ans += tmp * (tmp + 1L) / 2L;
				}
			}
			System.out.println(ans);
		} else {
			long ans = 1L * list.get(0) * N;
			System.out.println(ans * (ans + 1L) / 2L);
		}
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		String S = sc.next();
		Stack<Character> stack = new Stack<>();
		for (char asc : S.toCharArray()) {
			if (asc == '+' || asc == '-' || asc == '*' || asc == '/') {
				stack.push(asc);
			} else if (asc == ',') {
				System.out.print(stack.peek());
			} else if (asc == ')') {
				stack.pop();
				System.out.print(asc);
			} else {
				System.out.print(asc);
			}
		}
		System.out.println();
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long[] A = new long[N];
		long[] B = new long[N];
		for (int i = 0; i < N; i++) {
			A[i] = sc.nextLong();
			B[i] = A[i];
		}
		for (int i = 0; i < N; i++) {
			for (int left = 1; i - left >= 0; left++) {
				if (A[i - left] > B[i] + 1L * left * left) {
					A[i - left] = B[i] + 1L * left * left;
				} else if (B[i - left] < B[i] + 1L * left * left) {
					break;
				}
			}
			for (int right = 1; i + right < N; right++) {
				if (A[i + right] > B[i] + 1L * right * right) {
					A[i + right] = B[i] + 1L * right * right;
				} else if (B[i + right] <= B[i] + 1L * right * right){
					break;
				}
			}
		}
		for (int i = 0; i < N; i++) {
			System.out.println(A[i]);
		}
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		System.out.println(N);
	}

	private void solveE() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		System.out.println(N);
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

	interface BitSet {
		void set(int index, boolean bit);
		boolean get(int index);
		void shiftRight(int num);
		void shiftLeft(int num);
		void or(BitSet bitset);
		void and(BitSet bitset);
	}

	/**
	 * Longの配列によるBitSetの実装
	 * get/setはO(1)
	 * shift/or/andはO(size / 64)
	 */
	class LongBit implements BitSet {
		long[] bitArray;

		public LongBit(int size) {
			bitArray = new long[((size + 63) / 64)];
		}

		@Override
		public void set(int index, boolean bit) {
			int segment = index / 64;
			int inIndex = index % 64;
			if (bit) {
				bitArray[segment] |= 1L << inIndex;
			} else {
				bitArray[segment] &= ~(1L << inIndex);
			}
		}

		@Override
		public boolean get(int index) {
			int segment = index / 64;
			int inIndex = index % 64;
			return (bitArray[segment] & (1L << inIndex)) != 0L;
		}

		@Override
		public void shiftRight(int num) {
			int shiftSeg = num / 64;
			int shiftInI = num % 64;
			for (int segment = 0; segment < bitArray.length; segment++) {
				int sourceSeg = segment + shiftSeg;
				if (sourceSeg < bitArray.length) {
					bitArray[segment] = bitArray[sourceSeg] >>> shiftInI;
					if (shiftInI > 0 && sourceSeg + 1 < bitArray.length) {
						bitArray[segment] |= bitArray[sourceSeg + 1] << (64 - shiftInI);
					}
				} else {
					bitArray[segment] = 0L;
				}
			}
		}

		@Override
		public void shiftLeft(int num) {
			int shiftSeg = num / 64;
			int shiftInI = num % 64;
			for (int segment = bitArray.length - 1; segment >= 0; segment--) {
				int sourceSeg = segment - shiftSeg;
				if (sourceSeg >= 0) {
					bitArray[segment] = bitArray[sourceSeg] << shiftInI;
					if (shiftInI > 0 && sourceSeg > 0) {
						bitArray[segment] |= bitArray[sourceSeg - 1] >>> (64 - shiftInI);
					}
				} else {
					bitArray[segment] = 0L;
				}
			}
		}

		public long getLong(int segment) {
			return bitArray[segment];
		}

		@Override
		public void or(BitSet bitset) {
			if (!(bitset instanceof LongBit)) {
				return;
			}
			for (int segment = 0; segment < bitArray.length; segment++) {
				bitArray[segment] |= ((LongBit)bitset).getLong(segment);
			}
		}

		@Override
		public void and(BitSet bitset) {
			if (!(bitset instanceof LongBit)) {
				return;
			}
			for (int segment = 0; segment < bitArray.length; segment++) {
				bitArray[segment] &= ((LongBit)bitset).getLong(segment);
			}
		}
	}

}