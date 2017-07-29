package arc079;

import java.util.Scanner;

public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveC();
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int M = sc.nextInt();
		boolean[] a = new boolean[N + 1];
		boolean[] b = new boolean[N + 1];
		for (int i = 0; i < M; i++) {
			int a_i = sc.nextInt();
			int b_i = sc.nextInt();

			if (a_i == 1) {
				a[b_i] = true;
			}
			if (b_i == 1) {
				a[a_i] = true;
			}
			if (a_i == N) {
				b[b_i] = true;
			}
			if (b_i == N) {
				b[a_i] = true;
			}
		}
		boolean answer = false;
		for (int i = 1; i <= N; i++) {
			if (a[i] && b[i]) {
				answer = true;
				break;
			}
		}
		if (answer) {
			System.out.println("POSSIBLE");
		} else {
			System.out.println("IMPOSSIBLE");
		}
	}
}