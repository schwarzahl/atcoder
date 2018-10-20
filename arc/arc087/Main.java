package arc087;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

public class Main {
	public static void main(String[] args) {
		Main main = new Main();
		main.solveD();
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		String s = sc.next();
		sc.nextLine();
		int x = sc.nextInt();
		int y = sc.nextInt();

		List<Integer> f_list = new ArrayList<>();
		{
			int f_count = 0;
			for (int i = 0; i < s.length(); i++) {
				char c = s.charAt(i);
				if (c == 'F') {
					f_count++;
				}
				if (c == 'T') {
					f_list.add(f_count);
					f_count = 0;
				}
			}
			f_list.add(f_count);
		}

		Set<Integer> x_set = new HashSet<>();
		Set<Integer> y_set = new HashSet<>();

		x_set.add(f_list.get(0));
		y_set.add(0);

		for (int index = 1; index < f_list.size(); index++) {
			Set<Integer> next_set = new HashSet<>();
			int f = f_list.get(index);
			if (index % 2 == 0) {
				for (int tmp_x : x_set) {
					next_set.add(tmp_x + f);
					next_set.add(tmp_x - f);
				}
				x_set = next_set;
			} else {
				for (int tmp_y : y_set) {
					next_set.add(tmp_y + f);
					next_set.add(tmp_y - f);
				}
				y_set = next_set;
			}
		}
		if (x_set.contains(x) && y_set.contains(y)) {
			System.out.println("Yes");
		} else {
			System.out.println("No");
		}
	}
}