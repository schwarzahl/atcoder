package xmas_contest_2017;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Optional;
import java.util.Scanner;
import java.util.Set;
import java.util.StringJoiner;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveE();
	}

	private void solveA() {
		System.out.println("BBBAAABBAAAABAB");
	}

	private void solveB() {
		System.out.println("17 16");
		System.out.println("...............*");
		System.out.println("..............X7");
		System.out.println(".............m1.");
		System.out.println("............a0..");
		System.out.println("...........s2...");
		System.out.println("..........Ct....");
		System.out.println(".........os.....");
		System.out.println("........ne......");
		System.out.println(".......tt.......");
		System.out.println("......en........");
		System.out.println(".....so.........");
		System.out.println("....tC..........");
		System.out.println("...2s...........");
		System.out.println("..0a............");
		System.out.println(".1m.............");
		System.out.println("7X..............");
		System.out.println("*...............");
	}

	private void solveC() {
		// B = not A -> 255-A
		System.out.println("b = a 1000 a");
		// D = not 0 -> 255
		System.out.println("d = c 1000 c");
		// E = (255 - A) + 255 -> 254 - A
		System.out.println("e = b + d");
		// A = not E - > 255 - (254 - A) -> A + 1
		System.out.println("a = e 1000 e");
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		System.out.println(N);
	}

	private void solveE() {
		Scanner sc = new Scanner(System.in);
		char[] S = sc.next().toCharArray();
		char[] T = sc.next().toCharArray();
		boolean[][] memo = new boolean[1001][];
		for (int i = 0; i < 1001; i++) {
			memo[i] = new boolean[1001];
		}
		if (search(S, T, 0, 0, memo)) {
			System.out.println("YES");
		} else {
			System.out.println("NO");
		}
	}

	private boolean search(char[] S, char[] T, int s_index, int t_index, boolean[][] memo) {
		if (memo[s_index][t_index]) {
			return false;
		}
		memo[s_index][t_index] = true;
		if (s_index == S.length && t_index == T.length) {
			return true;
		}
		boolean ans = false;
		if (s_index < S.length && S[s_index] == 'A') {
			ans |= search(S, T, s_index + 1, t_index, memo);
		}
		if (t_index < T.length && T[t_index] == 'B') {
			ans |= search(S, T, s_index, t_index + 1, memo);
		}
		if (s_index < S.length && t_index < T.length && S[s_index] == T[t_index]) {
			ans |= search(S, T, s_index + 1, t_index + 1, memo);
		}
		return ans;
	}

	private void solveF() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		System.out.println(N);
	}

	private void solveG() {
		System.out.println("ACEFGHNPQSYZ");
	}

	private void solveH() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long M = sc.nextLong();
		long[] x = new long[N + 2];
		x[0] = 0L;
		x[1] = 1L;
		x[2] = 2L;
		x[3] = 4L;
		long add = 3L;
		int index = 0;
		for (int i = 3; i < N; i += 3) {
			while (add >= x[index]) {
				if (add == x[index]) {
					add++;
				} else {
					index++;
				}
			}
			x[i] = x[i-1] * 2;
			x[i+1] = x[i] + add;
			x[i+2] = x[i + 1] + add;
			add++;
		}
		{
			StringJoiner sj = new StringJoiner(" ");
			Arrays.stream(x).limit(N).forEach(asc -> sj.add(String.valueOf(asc)));
			System.out.printf("%s\n", sj.toString());
		}
		int Q = sc.nextInt();
		for (int i = 0; i < Q; i++) {
			long sum = sc.nextLong();
			int left = 0;
			int right = N - 1;
			while (true) {
				if (x[left] + x[right] < sum) {
					left++;
				} else if (x[left] + x[right] > sum) {
					right--;
				} else {
					break;
				}
			}
			System.out.printf("%s\n", left + " " + right);
		}
	}

	private void solveI() {
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
}