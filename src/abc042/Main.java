package abc042;

import java.util.Scanner;
public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveA();
	}

	private void solveA() {
		Scanner sc = new Scanner(System.in);
		int five = 0;
		int seven = 0;
		for (int i = 0; i < 3; i++) {
			switch (sc.nextInt()) {
			case 5:
				five++;
				break;
			case 7:
				seven++;
				break;
			default:
			}
		}
		if (five == 2 && seven == 1) {
			System.out.println("YES");
		} else {
			System.out.println("NO");
		}
	}
}