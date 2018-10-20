package agc013;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveC();
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		int L = sc.nextInt();
		int T = sc.nextInt();
		int offset = L;
		List<Integer> list = new ArrayList<Integer>();
		for (int i = 0; i < N; i++) {
			int x = sc.nextInt();
			int d = sc.nextInt();
			int g = x + T * (3 - d * 2);
			list.add(g);
			if (offset > g) {
				offset = g;
			}
		}
		List<Integer> relativeList = new ArrayList<Integer>();
		int index = 0;
		for (int x : list) {
			relativeList.add((x - offset) % L);
			if (x - offset >= L) {
				index += (x - offset) / L;
			}
		}
		Collections.sort(relativeList);
		index %= N;
		for (int x : relativeList.subList(index, N)) {
			System.out.println((((x + offset) % L) + L) % L);
		}
		for (int x : relativeList.subList(0, index)) {
			System.out.println((((x + offset) % L) + L) % L);
		}
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		if (N > 1) {
			int num = 1;
			int prev = sc.nextInt();
			Boolean isInc = null;
			for (int i = 1; i < N; i++) {
				int tmp = sc.nextInt();
				if (prev < tmp) {
					if (isInc == null) {
						isInc = true;
					} else if (!isInc) {
						num++;
						isInc = null;
					}
				}
				if (prev > tmp) {
					if (isInc == null) {
						isInc = false;
					} else if (isInc) {
						num++;
						isInc = null;
					}
				}
				prev = tmp;
			}
			System.out.println(num);
		} else {
			System.out.println(1);
		}
	}
}