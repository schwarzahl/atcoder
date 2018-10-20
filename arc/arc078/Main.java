package arc078;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

public class Main {
	public static void main(String[] args){
		Main main = new Main();
		main.solveD();
	}

	private void solveD() {
		Scanner sc = new Scanner(System.in);

		int N = sc.nextInt();
		Map<Integer, Set<Integer>> cMap = new HashMap<>();
		for (int i = 1; i <= N; i++) {
			cMap.put(i, new HashSet<>());
		}
		for (int i = 0; i < N - 1; i++) {
			int a = sc.nextInt();
			int b = sc.nextInt();
			cMap.get(a).add(b);
			cMap.get(b).add(a);
		}
		int[] parentMap = new int[N + 1];
		Map<Integer, Set<Integer>> childMap = new HashMap<>();
		for (int i = 1; i<= N; i++) {
			parentMap[i] = 0;
			childMap.put(i, new HashSet<>());
		}
		List<Integer> list = new ArrayList<>();
		list.add(1);
		int depth = 1;
		int sDepth = 0;
		while (!list.isEmpty()) {
			List<Integer> nextList = new ArrayList<>();
			for (int parent : list) {
				for (int i : cMap.get(parent)) {
					if (parentMap[parent] != i ) {
						parentMap[i] = parent;
						nextList.add(i);
						if (i == N) {
							sDepth = depth;
						}
						childMap.get(parent).add(i);
					}
				}
			}
			list = nextList;
			depth++;
		}
		int jNode = N;
		for (int i = 0; i < (sDepth - 1) / 2; i++) {
			jNode = parentMap[jNode];
		}
		int sNum = getChildNum(jNode, childMap);
		int fNum = N - 2 - sNum;
		if (fNum > sNum) {
			System.out.println("Fennec");
		} else {
			System.out.println("Snuke");
		}
	}

	private int getChildNum(int node, Map<Integer, Set<Integer>> childMap) {
		int sum = 0;
		for (int child : childMap.get(node)) {
			sum += getChildNum(child, childMap) + 1;
		}
		return sum;
	}

	private void solveC() {
		Scanner sc = new Scanner(System.in);

		int N = sc.nextInt();
		long[] a = new long[N];
		BigInteger sum = BigInteger.ZERO;
		for (int i = 0; i < N; i++) {
			a[i] = sc.nextLong();
			sum = sum.add(BigInteger.valueOf(a[i]));
		}
		sum = sum.subtract(BigInteger.valueOf(a[0]));
		BigInteger sum2 = BigInteger.valueOf(a[0]);
		BigInteger min = sum.subtract(sum2).abs();
		for (int i = 1; i < N - 1; i++) {
			BigInteger val = BigInteger.valueOf(a[i]);
			sum = sum.subtract(val);
			sum2 = sum2.add(val);
			BigInteger abs = sum.subtract(sum2).abs();
			if (abs.compareTo(min) < 0) {
				min = abs;
			}
		}
		System.out.println(min);
	}
}