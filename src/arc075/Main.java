package arc075;

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
		int sum = 0;
		List<Integer> list = new ArrayList<Integer>();
		for (int i = 0; i < N; i++) {
			int val = sc.nextInt();
			sum += val;
			list.add(val);
		}
		if (sum % 10 != 0) {
			System.out.println(sum);
		} else {
			Collections.sort(list);
			for (int val : list) {
				if (val % 10 != 0) {
					System.out.println(sum - val);
					return;
				}
			}
			System.out.println(0);
		}
	}
}