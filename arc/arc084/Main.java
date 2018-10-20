package arc084;

import java.util.Arrays;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveD();
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long[] aList = new long[N];
		long[] bList = new long[N];
		long[] cList = new long[N];
		for (int i = 0; i < N; i++) {
			aList[i] = sc.nextLong();
		}
		for (int i = 0; i < N; i++) {
			bList[i] = sc.nextLong();
		}
		for (int i = 0; i < N; i++) {
			cList[i] = sc.nextLong();
		}
		Arrays.sort(aList);
		Arrays.sort(bList);
		Arrays.sort(cList);
		int i_a = 0;
		int i_b = 0;
		int i_c = 0;
		long sum = 0L;
		for (; i_b < N; i_b++) {
			while (i_a < N && aList[i_a] < bList[i_b]) {
				i_a++;
			}
			while (i_c < N && bList[i_b] >= cList[i_c]) {
				i_c++;
			}
			sum+=((long)i_a)*((long)N-(long)i_c);
		}
		System.out.println(sum);
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		long K = sc.nextLong();
		int min = sum(K);
		long fin = Long.MAX_VALUE;
		for (long l = 1; l*K <= fin; l++) {
			int tmp = sum(l*K);
			fin = getNewFin(fin, tmp);
			if (min > tmp) {
				min = tmp;
			}
		}
		System.out.println(min);
	}

	private long getNewFin(long oldFin, int digit) {
		long newFin = 1;
		long curFin = oldFin;
		for (int i = 0; i < digit; i++) {
			curFin /= 10;
			newFin *= 10;
		}
		if (curFin > 0) {
			return newFin;
		} else {
			return oldFin;
		}
	}

	private int sum(long num) {
		int ret = 0;
		while (num > 0) {
			ret += num % 10;
			num /= 10;
		}
		return ret;
	}
}