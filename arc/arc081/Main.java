package arc081;

import java.util.Collections;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.TreeMap;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveC();
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();

		TreeMap<Integer, Integer> map = new TreeMap<>(Collections.reverseOrder());
		for (int i = 0; i < N; i++) {
			int a = sc.nextInt();
			if (map.containsKey(a)) {
				map.put(a, map.get(a) + 1);
			} else {
				map.put(a, 1);
			}
		}
		long max = 0;
		long secMax = 0;
		for (Entry<Integer, Integer> entry : map.entrySet()) {
			if (max == 0) {
				if (entry.getValue() > 3) {
					max = entry.getKey();
					secMax = entry.getKey();
				} else if (entry.getValue() > 1) {
					max = entry.getKey();
				}
			} else if (entry.getValue() > 1) {
				secMax = entry.getKey();
			}
			if (secMax > 0) {
				break;
			}
		}
		System.out.println(max * secMax);
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();

		System.out.println(N);
	}

	private void solveE() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();

		System.out.println(N);
	}
}