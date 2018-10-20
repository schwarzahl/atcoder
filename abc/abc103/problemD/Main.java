package abc103.problemD;

import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solve();
	}

	private void solve() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();
		int[] BtoA = new int[N + 1];
		for (int i = 1; i <= N; i++) {
			BtoA[i] = 0;
		}
		int[] A = new int[M + 1];
		int[] B = new int[M + 1];
		for (int i = 1; i <= M; i++) {
			A[i] = sc.nextInt();
			B[i] = sc.nextInt();
			if (BtoA[B[i]] < A[i]) {
				BtoA[B[i]] = A[i];
			}
		}
		int ans = 0;
		int maxRightCutId = 0;
		for (int i = 2; i <= N; i++) {
			if (maxRightCutId < BtoA[i]) {
				maxRightCutId = i - 1;
				ans++;
			}
		}
		System.out.println(ans);
	}
}