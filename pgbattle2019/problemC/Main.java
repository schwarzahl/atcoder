package problemC;

import java.io.IOException;
import java.io.InputStream;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solve();
	}

	private void solve() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int[] X = new int[N];
		int[] Y = new int[N];
		int[] D = new int[N];
		for (int i = 0; i < N; i++) {
			X[i] = sc.nextInt();
			Y[i] = sc.nextInt();
			D[i] = sc.nextInt();
		}
		BinaryIndexedTree2D bit = new BinaryIndexedTree2D(6003, 6003);
		for (int i = 0; i < N; i++) {
			bit.add(X[i] + Y[i] + 1, 3001 - X[i] + Y[i], 1);
		}
		for (int i = 0; i < N; i++) {
			int newX = X[i] + Y[i] + 1;
			int newY = 3001 - X[i] + Y[i];
			long ans = bit.getSum(Math.min(newX + D[i], 6002), Math.min(newY + D[i], 6002));
			ans += bit.getSum(Math.max(newX - D[i] - 1, 0), Math.max(newY - D[i] - 1, 0));
			ans -= bit.getSum(Math.max(newX - D[i] - 1, 0), Math.min(newY + D[i], 6002));
			ans -= bit.getSum(Math.min(newX + D[i], 6002), Math.max(newY - D[i] - 1, 0));
			System.out.println(ans);
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

	/**
	 * 1-indexedの2次元BIT配列
	 */
	class BinaryIndexedTree2D {
		private int[][] array;

		public BinaryIndexedTree2D(int size1, int size2) {
			this.array = new int[size1 + 1][];
			for (int i = 1; i <= size1; i++) {
				this.array[i] = new int[size2 + 1];
			}
		}

		/**
		 * 指定した要素に値を加算する
		 * 計算量はO(logN * logN)
		 * @param index1 加算する要素の1次元目の添字
		 * @param index2 加算する要素の2次元目の添字
		 * @param value 加算する量
		 */
		public void add(int index1, int index2, int value) {
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
		public int getSum(int index1, int index2) {
			int sum = 0;
			for (int i1 = index1; i1 > 0; i1 -= (i1 & -i1)) {
				for (int i2 = index2; i2 > 0; i2 -= (i2 & -i2)) {
					sum += array[i1][i2];
				}
			}
			return sum;
		}
	}
}