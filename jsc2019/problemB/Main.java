package problemB;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.IntStream;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solve();
	}

	private void solve() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long K = sc.nextLong();
		int[] A = new int[N];
		for (int i = 0; i < N; i++) {
			A[i] = sc.nextInt();
		}
		long[] left = new long[N];
		long[] right = new long[N];
		for (int base = 0; base < N; base++) {
			for (int compare = 0; compare < N; compare++) {
				if (base < compare && A[base] < A[compare]) {
					right[base]++;
				}
				if (base > compare && A[base] < A[compare]) {
					left[base]++;
				}
			}
		}
		long ans = 0L;
		long MOD = 1000000007L;
		for (int i = N - 1; i >= 0; i--) {
			ans = (ans + left[i] * (((1L + K) * K / 2L) % MOD)) % MOD;
			ans = (ans + right[i] * (((-1L + K) * K / 2L) % MOD)) % MOD;
		}
		System.out.println(ans);
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
            return (int) nextLong();
        }

        public double nextDouble() {
            return Double.parseDouble(next());
        }
    }
}