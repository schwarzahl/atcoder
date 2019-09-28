package problemD;

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
		long[] A = new long[N];
		for (int i = 0; i < N; i++) {
			A[i] = sc.nextLong();
		}
		long cost = 0L;
		for (int i = 1; i < N - 1; i++) {
			if (A[i - 1] < A[i] && A[i + 1] < A[i]) {
				long max = Math.max(A[i - 1], A[i + 1]);
				cost += A[i] - max;
				A[i] = max;
			}
			if (A[i - 1] > A[i] && A[i + 1] > A[i]) {
				long min = Math.min(A[i - 1], A[i + 1]);
				cost += min - A[i];
				A[i] = min;
			}
		}
		for (int i = 0; i < N - 1; i++) {
			cost += Math.abs(A[i] - A[i + 1]);
		}
		System.out.println(cost);
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
}