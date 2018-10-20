package abc076;

import java.util.Scanner;

public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveD();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		int R = sc.nextInt();
		int G = sc.nextInt();
		System.out.println(2 * G - R);
	}

	private void solveB() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int K = sc.nextInt();
		int output = 1;
		for (int i = 0; i < N; i++) {
			if (output < K) {
				output *= 2;
			} else {
				output += K;
			}
		}
		System.out.println(output);
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		String S = sc.next();
		String T = sc.next();
		int SL = S.length();
		int TL = T.length();
		for (int s_ind = SL - TL; s_ind >= 0; s_ind--) {
			int t_ind = 0;
			for (; t_ind < TL; t_ind++) {
				if (S.charAt(s_ind + t_ind) != '?' && S.charAt(s_ind + t_ind) != T.charAt(t_ind)) {
					break;
				}
			}
			if (t_ind == TL) {
				for (int o_ind = 0; o_ind < SL; o_ind++) {
					if (s_ind <= o_ind && o_ind < s_ind + TL) {
						System.out.print(T.charAt(o_ind - s_ind));
					} else {
						if (S.charAt(o_ind) == '?') {
							System.out.print('a');
						} else {
							System.out.print(S.charAt(o_ind));
						}
					}
				}
				System.out.println();
				return;
			}
		}
		System.out.println("UNRESTORABLE");
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int[] t = new int[N];
		int sum_t = 0;
		for (int i = 0; i < N; i++) {
			t[i] = sc.nextInt() * 2;
			sum_t += t[i];
		}
		int[] v = new int[N];
		for (int i = 0; i < N; i++) {
			v[i] = sc.nextInt() * 2;
		}
		int tmp_v = 0;
		int tmp_t = 0;
		int[] max_v = new int[sum_t + 1];
		for (int i = 0; i < N; i++) {
			max_v[tmp_t] = v[i] < tmp_v ? v[i] : tmp_v;
			tmp_v = max_v[tmp_t];
			for (int t_ind = 0; t_ind < t[i]; t_ind++) {
				max_v[++tmp_t] = v[i] < tmp_v + 1 ? v[i] : tmp_v + 1;
				tmp_v = max_v[tmp_t];
			}
		}
		tmp_v = 0;
		for (int i = N - 1; i >= 0; i--) {
			max_v[tmp_t] = v[i] < tmp_v ? v[i] : tmp_v;
			tmp_v = max_v[tmp_t];
			for (int t_ind = 0; t_ind < t[i]; t_ind++) {
				int back_v = v[i] < tmp_v + 1 ? v[i] : tmp_v + 1;
				--tmp_t;
				max_v[tmp_t] = max_v[tmp_t] < back_v ? max_v[tmp_t] : back_v;
				tmp_v = max_v[tmp_t];
			}
		}
		double answer = 0;
		for (int i = 0; i < sum_t; i++) {
			answer += (max_v[i] + max_v[i + 1]);
		}
		System.out.println(answer * 0.125);
	}
}