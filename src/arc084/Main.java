package arc084;

import java.util.Arrays;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveC();
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
			sum+=(i_a)*(N-i_c);
		}
		System.out.println(sum);
	}
}