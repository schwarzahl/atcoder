package arc075;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveE();
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

	private void solveD() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long A = sc.nextLong();
		long B = sc.nextLong();
		List<Integer> list = new ArrayList<Integer>();
		int max = 0;
		for (int i = 0; i < N; i++) {
			int val = sc.nextInt();
			list.add(val);
			if (max < val) {
				max = val;
			}
		}
		long minNum = (max + A - 1) / A;
		long maxNum = (max + B - 1) / B;
		do {
			long count = (minNum + maxNum) / 2;
			if (isWin(count, A, B, list)) {
				maxNum = count;
			} else {
				minNum = count + 1;
			}
		} while (minNum < maxNum);
		System.out.println(maxNum);
	}

	private boolean isWin(long count, long A, long B, List<Integer> hps) {
		long sum = 0;
		for (int hp : hps) {
			long num = (hp - count * B + A - B - 1) / (A - B);
			if (num > 0) {
				sum += num;
			}
		}
		return sum <= count;
	}

	private void solveE() {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		long K = sc.nextLong();
		long[] sumArr = new long[N + 1];
		sumArr[0] = 0L;
		List<Long> sumList = new ArrayList<Long>();
		sumList.add(sumArr[0]);
		for (int i = 1; i <= N; i++) {
			sumArr[i] = sumArr[i - 1] + sc.nextLong() - K;
			sumList.add(sumArr[i]);
		}
		Collections.sort(sumList);
		int current = 0;
		Map<Long, Integer> map = new HashMap<>();
		for (long value : sumList) {
			map.put(value, current++);
		}
		long ans = 0L;
		BinaryIndexedTree bit = new BinaryIndexedTree(N + 1);
		for (int i = 0; i <= N; i++) {
			int pos = map.get(sumArr[i]);
			ans += bit.getSum(pos);
			bit.addNum(1, pos);
		}
		System.out.println(ans);
	}

	class BinaryIndexedTree {
		int N;
		int[] arr;
		public BinaryIndexedTree(int N) {
			arr = new int[N];
			this.N = N;
		}

		/**
		 * posで指定した位置にval加算する
		 * @param val 加算する値
		 * @param pos 加算する位置
		 */
		public void addNum(int val, int pos) {
			while (pos < N) {
				arr[pos] += val;
				pos += (pos+1)&(-pos-1);
			}
		}

		/**
		 * posで指定した位置までの和を取得する
		 * @param pos 和を取得する位置
		 * @return 0～posの和
		 */
		public int getSum(int pos) {
			int sum = 0;
			while (pos >= 0) {
				sum += arr[pos];
				pos -= (pos+1)&(-pos-1);
			}
			return sum;
		}
	}
}