class Solution:
    def __init__(self):
        pass

    def baekjoon1000(self):
        # 두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.
        # 첫째 줄에 A와 B가 주어진다. (0 < A, B < 10)
        # 첫째 줄에 A+B를 출력한다.
        A, B = map(int, input().split())
        print(A + B)
    
    def baekjoon1001(self):
        # 문제
        # 두 정수 A와 B를 입력받은 다음, A-B를 출력하는 프로그램을 작성하시오.

        # 입력
        # 첫째 줄에 A와 B가 주어진다. (0 < A, B < 10)

        # 출력
        # 첫째 줄에 A-B를 출력한다.
        A, B = map(int, input().split())
        print(A - B)
    
    def baekjoon1002(self):
        # 첫째 줄에 테스트 케이스의 개수 T가 주어진다. 
        # 각 테스트 케이스는 다음과 같이 이루어져 있다.
        # 한 줄에 x1, y1, r1, x2, y2, r2가 주어진다. 
        # x1, y1, x2, y2는 -10,000보다 크거나 같고, 
        # 10,000보다 작거나 같은 정수이고, 
        # r1, r2는 10,000보다 작거나 같은 음이 아닌 정수이다.

        T = int(input())

        for i in range(T):
            x1, y1, r1, x2, y2, r2 = map(int, input().split())

            # 두 점 사이의 거리 공식 (https://m.blog.naver.com/galaxyenergy/221263626715)
            d = ((x1 - x2)**2 + (y1 -y2)**2)**(1/2)
            

if __name__ == "__main__":
    s = Solution()
    s.baekjoon1001()