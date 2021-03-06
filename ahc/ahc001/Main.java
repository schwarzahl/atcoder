import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.stream.IntStream;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solve();
	}

	private void solve() {
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		Space[] spaces = new Space[n];
		Map<Integer, Space> id2Space = new HashMap<>();
		BitSet map = new LongBit(10000 * 10000);
		Queue<Space> queue = new PriorityQueue<>(Comparator.comparingInt(s -> (int)Math.round(1000000 * s.getScore())));
		for (int id = 0; id < n; id++) {
			int x = sc.nextInt();
			int y = sc.nextInt();
			int r = sc.nextInt();
			spaces[id] = new Space(id, x, y, r);
			id2Space.put(id, spaces[id]);
			queue.add(spaces[id]);
			map.set(x, y, true);
		}
		long start_time = System.currentTimeMillis();
		while (!queue.isEmpty()) {
			Space space = queue.poll();
			if (space.getArea() >= space.r) {
				continue;
			}
			boolean next = false;
			for (int dir = 0; dir < 4; dir++) {
				next |= space.expand(dir, map);
			}
			if (next) {
				queue.add(space);
			}
		}
		Random rand = new Random();
		while (System.currentTimeMillis() - start_time < 3500) {
			Space space = spaces[rand.nextInt(n)];
			if (space.r <= space.getArea()) {
				int coDir = rand.nextInt(4);
				if (space.contract(coDir, map) && space.r > space.getArea()) {
					while (!space.expand(rand.nextInt(4), map)) ;
				}
			} else {
				space.expand(rand.nextInt(4), map);
			}
		}
		int[][] map2id = new int[10000][10000];
		for (Space space : spaces) {
			for (int x = space.left; x < space.right; x++) {
				for (int y = space.top; y < space.bottom; y++) {
					map2id[x][y] = space.id + 1;
				}
			}
		}
		Arrays.sort(spaces, Comparator.comparingInt(s -> (int)Math.round(1000000 * s.getScore())));
		for (Space space : spaces) {
			for (int dir = 0; dir < 4; dir++) {
				Set<Integer> idSet = space.expandCheck(dir, map2id);
				if (idSet == null) continue;
				double before = space.getTrueScore();
				double after = space.getTrueScore(dir, +1);
				for (int id : idSet) {
					Space target = id2Space.get(id);
					before += target.getTrueScore();
					after += target.getTrueScore((dir + 2) % 4, -1);
				}
				if (before < after) {
					space.expand(dir, map2id);
					for (int id : idSet) {
						Space target = id2Space.get(id);
						target.contract((dir + 2) % 4, map2id);
					}
					dir--;
				}
			}
		}
		Arrays.sort(spaces, Comparator.comparingInt(s -> s.id));
		for (Space space : spaces) {
			System.out.println(String.format("%d %d %d %d", space.left, space.top, space.right, space.bottom));
		}
	}

	class Space {
		int id;
		int x;
		int y;
		int r;
		int left;
		int top;
		int right;
		int bottom;
		public Space(int id, int x, int y, int r) {
			this.id = id;
			this.x = x;
			this.y = y;
			this.r = r;
			left = x;
			top = y;
			right = x + 1;
			bottom = y + 1;
		}

		public int getArea() {
			return (right - left) * (bottom - top);
		}
		public int getArea(int dir, int mode) {
			int height = bottom - top + (dir % 2 == 1 ? mode : 0);
			int width = right - left + (dir % 2 == 0 ? mode : 0);
			return height * width;
		}
		public double getScore() {
			return 1.0 * Math.min(r, getArea()) / Math.max(r, getArea());
		}
		public double getTrueScore() {
			double tmp = 1.0 - getScore();
			return 1.0 - tmp * tmp;
		}
		public double getTrueScore(int dir, int mode) {
			int left = this.left - (dir == 0 ? mode : 0);
			int top = this.top - (dir == 1 ? mode : 0);
			int right = this.right + (dir == 2 ? mode : 0);
			int bottom = this.bottom + (dir == 3 ? mode : 0);
			if (this.x < left || right <= this.x || this.y < top || bottom <= this.y) {
				return -10000.0;
			}
			int area = (bottom - top) * (right - left);
			double tmp = 1.0 - 1.0 * Math.min(this.r, area) / Math.max(this.r, area);
			return 1.0 - tmp * tmp;
		}
		public boolean expand(int dir, BitSet map) {
			if (dir == 0) {
				// left
				if (left - 1 < 0) {
					return false;
				}
				if (map.getRangeOr(left - 1, top, bottom)) {
					return false;
				}
				map.setRange(left - 1, top, bottom, true);
				left--;
				return true;
			}
			if (dir == 1) {
				// top
				if (top - 1 < 0) {
					return false;
				}
				for (int x = left; x < right; x++) {
					if (map.get(x, top - 1)) {
						return false;
					}
				}
				for (int x = left; x < right; x++) {
					map.set(x, top - 1, true);
				}
				top--;
				return true;
			}
			if (dir == 2) {
				// right
				if (right + 1 > 10000) {
					return false;
				}
				if (map.getRangeOr(right, top, bottom)) {
					return false;
				}
				map.setRange(right, top, bottom, true);
				right++;
				return true;
			}
			if (dir == 3) {
				// bottom
				if (bottom + 1 > 10000) {
					return false;
				}
				for (int x = left; x < right; x++) {
					if (map.get(x, bottom)) {
						return false;
					}
				}
				for (int x = left; x < right; x++) {
					map.set(x, bottom, true);
				}
				bottom++;
				return true;
			}
			// no reach
			return false;
		}
		public Set<Integer> expandCheck(int dir, int[][] map) {
			Set<Integer> set = new HashSet<>();
			if (dir == 0) {
				// left
				if (left - 1 < 0) {
					return null;
				}
				for (int y = top; y < bottom; y++) {
					if (map[left - 1][y] > 0) {
						set.add(map[left - 1][y] - 1);
					}
				}
			}
			if (dir == 1) {
				// top
				if (top - 1 < 0) {
					return null;
				}
				for (int x = left; x < right; x++) {
					if (map[x][top - 1] > 0) {
						set.add(map[x][top - 1] - 1);
					}
				}
			}
			if (dir == 2) {
				// right
				if (right + 1 > 10000) {
					return null;
				}
				for (int y = top; y < bottom; y++) {
					if (map[right][y] > 0) {
						set.add(map[right][y] - 1);
					}
				}
			}
			if (dir == 3) {
				// bottom
				if (bottom + 1 > 10000) {
					return null;
				}
				for (int x = left; x < right; x++) {
					if (map[x][bottom] > 0) {
						set.add(map[x][bottom] - 1);
					}
				}
			}
			return set;
		}
		public void expand(int dir, int[][] map) {
			if (dir == 0) {
				// left
				for (int y = top; y < bottom; y++) {
					map[left - 1][y] = id + 1;
				}
				left--;
			}
			if (dir == 1) {
				// top
				for (int x = left; x < right; x++) {
					map[x][top - 1] = id + 1;
				}
				top--;
			}
			if (dir == 2) {
				// right
				for (int y = top; y < bottom; y++) {
					map[right][y] = id + 1;
				}
				right++;
			}
			if (dir == 3) {
				// bottom
				for (int x = left; x < right; x++) {
					map[x][bottom] = id + 1;
				}
				bottom++;
			}
		}
		public boolean contract(int dir, BitSet map) {
			if (dir == 0) {
				// left
				if (left + 1 > x) {
					return false;
				}
				map.setRange(left, top, bottom, false);
				left++;
				return true;
			}
			if (dir == 1) {
				// top
				if (top + 1 > y) {
					return false;
				}
				for (int x = left; x < right; x++) {
					map.set(x, top, false);
				}
				top++;
				return true;
			}
			if (dir == 2) {
				// right
				if (right - 1 <= x) {
					return false;
				}
				map.setRange(right - 1, top, bottom, false);
				right--;
				return true;
			}
			if (dir == 3) {
				// bottom
				if (bottom - 1 <= y) {
					return false;
				}
				for (int x = left; x < right; x++) {
					map.set(x, bottom - 1, false);
				}
				bottom--;
				return true;
			}
			// no reach
			return false;
		}
		public void contract(int dir, int[][] map) {
			if (dir == 0) {
				// left
				for (int y = top; y < bottom; y++) {
					if (map[left][y] == id + 1) {
						map[left][y] = 0;
					}
				}
				left++;
			}
			if (dir == 1) {
				// top
				for (int x = left; x < right; x++) {
					if (map[x][top] == id + 1) {
						map[x][top] = 0;
					}
				}
				top++;
			}
			if (dir == 2) {
				// right
				for (int y = top; y < bottom; y++) {
					if (map[right - 1][y] == id + 1) {
						map[right - 1][y] = 0;
					}
				}
				right--;
			}
			if (dir == 3) {
				// bottom
				for (int x = left; x < right; x++) {
					if (map[x][bottom - 1] == id + 1) {
						map[x][bottom - 1] = 0;
					}
				}
				bottom--;
			}
		}
	}

	class Scanner {
		private InputStream in;
		private byte[] buffer = new byte[1024];
		private int index;
		private int length;

		public Scanner(InputStream in) {
			this.in = in;
		}

		private boolean isPrintableChar(int c) {
			return '!' <= c && c <= '~';
		}

		private boolean isDigit(int c) {
			return '0' <= c && c <= '9';
		}

		private boolean hasNextByte() {
			if (index < length) {
				return true;
			} else {
				try {
					length = in.read(buffer);
					index = 0;
				} catch (IOException e) {
					e.printStackTrace();
				}
				return length > 0;
			}
		}

		private boolean hasNext() {
			while (hasNextByte() && !isPrintableChar(buffer[index])) {
				index++;
			}
			return hasNextByte();
		}

		private int readByte() {
			return hasNextByte() ? buffer[index++] : -1;
		}

		public String next() {
			if (!hasNext()) {
				throw new RuntimeException("no input");
			}
			StringBuilder sb = new StringBuilder();
			int b = readByte();
			while (isPrintableChar(b)) {
				sb.appendCodePoint(b);
				b = readByte();
			}
			return sb.toString();
		}

		public long nextLong() {
			if (!hasNext()) {
				throw new RuntimeException("no input");
			}
			long value = 0L;
			boolean minus = false;
			int b = readByte();
			if (b == '-') {
				minus = true;
				b = readByte();
			}
			while (isPrintableChar(b)) {
				if (isDigit(b)) {
					value = value * 10 + (b - '0');
				}
				b = readByte();
			}
			return minus ? -value : value;
		}

		public int nextInt() {
			return (int)nextLong();
		}

		public double nextDouble() {
			return Double.parseDouble(next());
		}
	}

	interface CombCalculator {
		long comb(int n, int m);
	}

	interface MobiusFunction {
		int get(int n);
	}

	/**
	 * メビウス関数をエラトステネスの篩っぽく前計算するクラスです。
	 * 計算量はO(1)で、前計算でO(N logN)です。
	 */
	class SieveMobiusFunction implements MobiusFunction {
		int size;
		int[] mobiusFunctionValues;

		public SieveMobiusFunction(int size) {
			this.size = size;
			mobiusFunctionValues = new int[size];

			mobiusFunctionValues[0] = 0;
			mobiusFunctionValues[1] = 1;
			for (int i = 2; i < size; i++) {
				mobiusFunctionValues[i] = 1;
			}
			for (int i = 2; i * i < size; i++) {
				for (int k = 1; i * i * k < size; k++) {
					mobiusFunctionValues[i * i * k] *= 0;
				}
			}

			for (int i = 2; i < size; i++) {
				if (mobiusFunctionValues[i] == 1) {
					for (int k = 1; i * k < size; k++) {
						mobiusFunctionValues[i * k] *= -2;
					}
				}
				if (mobiusFunctionValues[i] > 1) {
					mobiusFunctionValues[i] = 1;
				}
				if (mobiusFunctionValues[i] < -1) {
					mobiusFunctionValues[i] = -1;
				}
			}
		}

		@Override
		public int get(int n) {
			if (n > size) {
				throw new RuntimeException("n is greater than size.");
			}
			if (n < 0) {
				return 0;
			}
			return mobiusFunctionValues[n];
		}
	}

	/**
	 * メビウス関数を定義通り計算するクラスです。
	 * 計算量はO(logN)です。
	 */
	class PrimeFactorizationMobiusFunction implements MobiusFunction {
		@Override
		public int get(int n) {
			if (n <= 0) {
				return 0;
			}
			if (n == 1) {
				return 1;
			}
			int num = 0;
			for (int i = 2; i < n; i++) {
				if (n % i == 0) {
					n /= i;
					num++;
					if (n % i == 0) {
						return 0;
					}
				}
			}
			return num % 2 == 0 ? -1 : 1;
		}
	}

	/**
	 * 組み合わせ計算を階乗の値で行うクラスです(MOD対応)
	 * 階乗とその逆元は前計算してテーブルに格納します。
	 * C(N, N) % M の計算量は O(1)、 前計算でO(max(N, logM))です。
	 * sizeを1e8より大きい値で実行するとMLEの危険性があります。
	 * また素数以外のMODには対応していません(逆元の計算に素数の剰余環の性質を利用しているため)。
	 */
	class FactorialTableCombCalculator implements CombCalculator {
		int size;
		long[] factorialTable;
		long[] inverseFactorialTable;
		long mod;

		public FactorialTableCombCalculator(int size, long mod) {
			this.size = size;
			factorialTable = new long[size + 1];
			inverseFactorialTable = new long[size + 1];
			this.mod = mod;

			factorialTable[0] = 1L;
			for (int i = 1; i <= size; i++) {
				factorialTable[i] = (factorialTable[i - 1] * i) % mod;
			}
			inverseFactorialTable[size] = inverse(factorialTable[size], mod);
			for (int i = size - 1; i >= 0; i--) {
				inverseFactorialTable[i] = (inverseFactorialTable[i + 1] * (i + 1)) % mod;
			}
		}

		private long inverse(long n, long mod) {
			return pow(n, mod - 2, mod);
		}

		private long pow(long n, long p, long mod) {
			if (p == 0) {
				return 1L;
			}
			long half = pow(n, p / 2, mod);
			long ret = (half * half) % mod;
			if (p % 2 == 1) {
				ret = (ret * n) % mod;
			}
			return ret;
		}

		@Override
		public long comb(int n, int m) {
			if (n > size) {
				throw new RuntimeException("n is greater than size.");
			}
			if (n < 0 || m < 0 || n < m) {
				return 0L;
			}
			return (((factorialTable[n] * inverseFactorialTable[m]) % mod) * inverseFactorialTable[n - m]) % mod;
		}
	}

	/**
	 * 組み合わせ計算をテーブルで実装したクラスです(MOD対応)
	 * 前計算でO(N^2), combはO(1)で実行できます
	 * sizeを2 * 1e4より大きい値で実行するとMLEの危険性があります
	 */
	class TableCombCalculator implements CombCalculator {
		long[][] table;
		int size;

		public TableCombCalculator(int size, long mod) {
			this.size = size;
			table = new long[size + 1][];

			table[0] = new long[1];
			table[0][0] = 1L;
			for (int n = 1; n <= size; n++) {
				table[n] = new long[n + 1];
				table[n][0] = 1L;
				for (int m = 1; m < n; m++) {
					table[n][m] = (table[n - 1][m - 1] + table[n - 1][m]) % mod;
				}
				table[n][n] = 1L;
			}
		}

		@Override
		public long comb(int n, int m) {
			if (n > size) {
				throw new RuntimeException("n is greater than size.");
			}
			if (n < 0 || m < 0 || n < m) {
				return 0L;
			}
			return table[n][m];
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
		 * @param index 和の終端となる要素の添字
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

	/**
	 * 1-indexedの2次元BIT配列
	 */
	class BinaryIndexedTree2D {
		private long[][] array;

		public BinaryIndexedTree2D(int size1, int size2) {
			this.array = new long[size1 + 1][];
			for (int i = 1; i <= size1; i++) {
				this.array[i] = new long[size2 + 1];
			}
		}

		/**
		 * 指定した要素に値を加算する
		 * 計算量はO(logN * logN)
		 * @param index1 加算する要素の1次元目の添字
		 * @param index2 加算する要素の2次元目の添字
		 * @param value 加算する量
		 */
		public void add(int index1, int index2, long value) {
			for (int i1 = index1; i1 < array.length; i1 += (i1 & -i1)) {
				for (int i2 = index2; i2 < array.length; i2 += (i2 & -i2)) {
					array[i1][i2] += value;
				}
			}
		}

		/**
		 * (1,1)〜指定した要素までの矩形和を取得する
		 * 計算量はO(logN * logN)
		 * @param index1 和の終端となる要素の1次元目の添字
		 * @param index2 和の終端となる要素の2次元目の添字
		 * @return (1,1)〜(index1,index2)までの矩形和
		 */
		public long getSum(int index1, int index2) {
			long sum = 0L;
			for (int i1 = index1; i1 > 0; i1 -= (i1 & -i1)) {
				for (int i2 = index2; i2 > 0; i2 -= (i2 & -i2)) {
					sum += array[i1][i2];
				}
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
		void set(int x, int y, boolean bit);
		void setRange(int x, int start_y, int end_y, boolean bit);
		boolean get(int index);
		boolean get(int x, int y);
		boolean getRangeOr(int x, int start_y, int end_y);
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
		public void set(int x, int y, boolean bit) {
			set(x * 10000 + y, bit);
		}
		@Override
		public void setRange(int x, int start_y, int end_y, boolean bit) {
			int start_index = x * 10000 + start_y;
			int end_index = x * 10000 + end_y;
			for (int index = start_index; index < end_index; index++) {
				if (index % 64 == 0 && end_index - index > 64) {
					bitArray[index / 64] = bit ? -1 : 0;
					index += 63;
				} else {
					set(index, bit);
				}
			}
		}

		@Override
		public boolean get(int index) {
			int segment = index / 64;
			int inIndex = index % 64;
			return (bitArray[segment] & (1L << inIndex)) != 0L;
		}
		@Override
		public boolean get(int x, int y) {
			return get(x * 10000 + y);
		}
		@Override
		public boolean getRangeOr(int x, int start_y, int end_y) {
			int start_index = x * 10000 + start_y;
			int end_index = x * 10000 + end_y;
			for (int index = start_index; index < end_index; index++) {
				if (index % 64 == 0 && end_index - index > 64) {
					if (bitArray[index / 64] != 0) {
						return true;
					}
					index += 63;
				} else if (get(index)) {
					return true;
				}
			}
			return false;
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