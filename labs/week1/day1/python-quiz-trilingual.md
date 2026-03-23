# Python 기초 사전 퀴즈 / Python Basic Pre-Quiz / Bài kiểm tra Python cơ bản

총 40문제 / 40 Questions / 40 Câu hỏi

---

## Section 1: 변수와 자료형 / Variables & Data Types / Biến và kiểu dữ liệu

---

**Q1.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
x = 10
x = "hello"
print(type(x))
```
- A) `<class 'int'>`
- B) `<class 'str'>`
- C) Error / 에러 / Lỗi
- D) `<class 'NoneType'>`

---

**Q2.** 다음 중 유효한 변수명이 아닌 것은? / Which is NOT a valid variable name? / Tên biến nào KHÔNG hợp lệ?
- A) `my_var`
- B) `_count`
- C) `2nd_value`
- D) `firstName`

---

**Q3.** `type(3.14)`의 결과는? / What does `type(3.14)` return? / `type(3.14)` trả về gì?
- A) `<class 'int'>`
- B) `<class 'float'>`
- C) `<class 'double'>`
- D) `<class 'decimal'>`

---

**Q4.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
a = True
b = False
print(a + b)
```
- A) `TrueFalse`
- B) `1`
- C) `True`
- D) Error / 에러 / Lỗi

---

**Q5.** `None`에 대한 올바른 설명은? / Which is correct about `None`? / Mô tả đúng về `None` là gì?
- A) `None`은 `0`과 같다 / `None` equals `0` / `None` bằng `0`
- B) `None`은 빈 문자열 `""`과 같다 / `None` equals `""` / `None` bằng `""`
- C) `None`은 값이 없음을 나타내는 고유한 타입이다 / `None` is a unique type meaning "no value" / `None` là kiểu riêng biệt nghĩa là "không có giá trị"
- D) `None`은 `False`와 동일하다 / `None` is the same as `False` / `None` giống với `False`

---

## Section 2: 문자열 / Strings / Chuỗi

---

**Q6.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
s = "Python"
print(s[0] + s[-1])
```
- A) `Pn`
- B) `Py`
- C) `on`
- D) Error / 에러 / Lỗi

---

**Q7.** `"hello world"`를 `"HELLO WORLD"`로 바꾸는 메서드는? / Which method converts to uppercase? / Phương thức nào chuyển thành chữ hoa?
- A) `s.capitalize()`
- B) `s.upper()`
- C) `s.title()`
- D) `s.swapcase()`

---

**Q8.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
name = "Alice"
print(f"Hello, {name}!")
```
- A) `Hello, {name}!`
- B) `Hello, Alice!`
- C) Error / 에러 / Lỗi
- D) `Hello, "Alice"!`

---

**Q9.** `"apple,banana,cherry".split(",")` 의 결과는? / What is the result? / Kết quả là gì?
- A) `"apple banana cherry"`
- B) `["apple", "banana", "cherry"]`
- C) `("apple", "banana", "cherry")`
- D) `{"apple", "banana", "cherry"}`

---

**Q10.** 여러 줄 문자열을 만드는 올바른 방법은? / Which creates a multiline string? / Cách nào tạo chuỗi nhiều dòng?
- A) `"line1\nline2"`
- B) `"""line1\nline2"""`
- C) A, B 모두 / Both A and B / Cả A và B
- D) 둘 다 아님 / Neither / Không cái nào

---

## Section 3: 리스트, 튜플, 딕셔너리 / List, Tuple, Dict / Danh sách, Tuple, Từ điển

---

**Q11.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
fruits = ["apple", "banana", "cherry"]
fruits.append("date")
print(len(fruits))
```
- A) `3`
- B) `4`
- C) `5`
- D) Error / 에러 / Lỗi

---

**Q12.** `[1, 2, 3, 4, 5][1:4]`의 결과는? / What is the result? / Kết quả là gì?
- A) `[1, 2, 3, 4]`
- B) `[2, 3, 4]`
- C) `[2, 3, 4, 5]`
- D) `[1, 2, 3]`

---

**Q13.** 튜플과 리스트의 차이점은? / What is the difference between tuple and list? / Sự khác biệt giữa tuple và list là gì?
- A) 튜플은 수정 불가 / Tuple is immutable / Tuple không thể thay đổi
- B) 튜플은 `[]`로 만든다 / Tuple uses `[]` / Tuple dùng `[]`
- C) 리스트는 수정 불가 / List is immutable / List không thể thay đổi
- D) 차이 없음 / No difference / Không có sự khác biệt

---

**Q14.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
d = {"name": "Alice", "age": 30}
print(d["name"])
```
- A) `Alice`
- B) `"Alice"`
- C) `name`
- D) Error / 에러 / Lỗi

---

**Q15.** 존재하지 않는 키를 안전하게 조회하는 방법은? / How to safely access a missing key? / Cách truy cập key không tồn tại an toàn?
```python
d = {"a": 1}
```
- A) `d["b"]`
- B) `d.get("b")`
- C) `d.find("b")`
- D) `d.search("b")`

---

## Section 4: 조건문과 반복문 / Control Flow / Luồng điều khiển

---

**Q16.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
x = 15
if x > 20:
    print("big")
elif x > 10:
    print("medium")
else:
    print("small")
```
- A) `big`
- B) `medium`
- C) `small`
- D) `big medium`

---

**Q17.** `for i in range(3)`에서 `i`의 값은? / What values does `i` take? / `i` nhận những giá trị nào?
- A) `1, 2, 3`
- B) `0, 1, 2`
- C) `0, 1, 2, 3`
- D) `1, 2`

---

**Q18.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
for i in range(5):
    if i == 3:
        break
    print(i, end=" ")
```
- A) `0 1 2 3 4`
- B) `0 1 2`
- C) `0 1 2 3`
- D) `0 1 2 4`

---

**Q19.** `continue` 키워드의 역할은? / What does `continue` do? / `continue` làm gì?
- A) 반복문을 완전히 종료 / Exits the loop entirely / Thoát hoàn toàn vòng lặp
- B) 현재 반복을 건너뛰고 다음으로 / Skips current iteration, goes to next / Bỏ qua vòng lặp hiện tại, chuyển sang tiếp theo
- C) 프로그램을 종료 / Exits the program / Thoát chương trình
- D) 함수를 종료 / Exits the function / Thoát hàm

---

**Q20.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
nums = [1, 2, 3, 4, 5]
result = [x * 2 for x in nums if x > 2]
print(result)
```
- A) `[2, 4, 6, 8, 10]`
- B) `[6, 8, 10]`
- C) `[3, 4, 5]`
- D) `[1, 2, 3, 4, 5]`

---

## Section 5: 함수 / Functions / Hàm

---

**Q21.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
def add(a, b=10):
    return a + b

print(add(5))
```
- A) `5`
- B) `10`
- C) `15`
- D) Error / 에러 / Lỗi

---

**Q22.** `*args`의 역할은? / What does `*args` do? / `*args` làm gì?
```python
def func(*args):
    print(args)
```
- A) 키워드 인자를 딕셔너리로 받음 / Receives keyword args as dict / Nhận đối số keyword dạng dict
- B) 위치 인자를 튜플로 받음 / Receives positional args as tuple / Nhận đối số vị trí dạng tuple
- C) 하나의 인자만 받음 / Receives only one arg / Chỉ nhận một đối số
- D) 리스트를 반환 / Returns a list / Trả về list

---

**Q23.** `**kwargs`의 역할은? / What does `**kwargs` do? / `**kwargs` làm gì?
- A) 위치 인자를 튜플로 받음 / Receives positional args as tuple / Nhận đối số vị trí dạng tuple
- B) 키워드 인자를 딕셔너리로 받음 / Receives keyword args as dict / Nhận đối số keyword dạng dict
- C) 두 개의 인자를 받음 / Receives exactly two args / Nhận đúng hai đối số
- D) Error / 에러 / Lỗi

---

**Q24.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
def greet(name):
    return f"Hi, {name}"

result = greet("Bob")
print(result)
```
- A) `None`
- B) `Hi, Bob`
- C) `greet("Bob")`
- D) Error / 에러 / Lỗi

---

**Q25.** lambda 함수의 올바른 사용은? / Which is correct lambda usage? / Cách dùng lambda đúng là?
- A) `lambda x: x * 2`
- B) `lambda: x * 2`
- C) `def lambda(x): return x * 2`
- D) `lambda x = x * 2`

---

## Section 6: 예외 처리 및 기타 / Exception Handling & More / Xử lý ngoại lệ & khác

---

**Q26.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
try:
    x = 1 / 0
except ZeroDivisionError:
    print("error")
finally:
    print("done")
```
- A) `error`
- B) `done`
- C) `error` 그리고 / and / và `done`
- D) 출력 없음 / No output / Không có kết quả

---

**Q27.** 파일을 열고 자동으로 닫는 구문은? / Which auto-closes a file? / Cách nào tự động đóng file?
- A) `open("file.txt")`
- B) `with open("file.txt") as f:`
- C) `file.open("file.txt")`
- D) `try: open("file.txt")`

---

**Q28.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
numbers = [3, 1, 4, 1, 5]
numbers.sort()
print(numbers)
```
- A) `[3, 1, 4, 1, 5]`
- B) `[1, 1, 3, 4, 5]`
- C) `[5, 4, 3, 1, 1]`
- D) Error / 에러 / Lỗi

---

**Q29.** `import`에 대한 올바른 설명은? / Which is correct about `import`? / Mô tả đúng về `import` là gì?
```python
import os
from datetime import datetime
```
- A) `import os`는 os 모듈 전체를 가져옴 / imports the entire os module / import toàn bộ module os
- B) `from datetime import datetime`은 datetime 클래스만 가져옴 / imports only the datetime class / chỉ import class datetime
- C) A, B 모두 맞음 / Both A and B are correct / Cả A và B đều đúng
- D) 둘 다 틀림 / Both are wrong / Cả hai đều sai

---

**Q30.** 다음 코드의 출력은? / What is the output? / Kết quả đầu ra là gì?
```python
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        return f"{self.name} says Woof!"

d = Dog("Rex")
print(d.bark())
```
- A) `Dog says Woof!`
- B) `Rex says Woof!`
- C) `Woof!`
- D) Error / 에러 / Lỗi

---

## Section 7: async / await 기초 / async / await Basics / async / await Cơ bản

---

**Q31.** `async def`로 정의한 함수를 무엇이라 부르는가? / What is an `async def` function called? / Hàm `async def` được gọi là gì?
- A) 일반 함수 / Normal function / Hàm thông thường
- B) 코루틴 / Coroutine / Coroutine
- C) 제너레이터 / Generator / Generator
- D) 데코레이터 / Decorator / Decorator

---

**Q32.** 다음 코드에서 `await`의 역할은? / What does `await` do here? / `await` làm gì ở đây?
```python
async def get_data():
    result = await fetch_from_db()
    return result
```
- A) 프로그램 전체를 멈춤 / Stops the entire program / Dừng toàn bộ chương trình
- B) 기다리되, 다른 작업이 실행될 수 있게 함 / Waits, but allows other tasks to run / Chờ, nhưng cho phép các tác vụ khác chạy
- C) 함수를 취소함 / Cancels the function / Hủy hàm
- D) 함수를 무시함 / Ignores the function / Bỏ qua hàm

---

**Q33.** `async def` 안에서만 사용할 수 있는 키워드는? / Which keyword works only inside `async def`? / Từ khóa nào chỉ dùng được trong `async def`?
- A) `return`
- B) `print`
- C) `await`
- D) `for`

---

**Q34.** 비동기(async)가 유용한 상황은? / When is async most useful? / Khi nào async hữu ích nhất?
- A) 복잡한 수학 계산 / Complex math calculations / Tính toán phức tạp
- B) DB 조회, API 호출 등 기다리는 시간이 많은 작업 / DB queries, API calls with lots of waiting time / Truy vấn DB, gọi API có nhiều thời gian chờ
- C) 파일 이름 바꾸기 / Renaming files / Đổi tên file
- D) 리스트 정렬 / Sorting a list / Sắp xếp danh sách

---

## Section 8: yield와 제너레이터 / yield & Generator / yield & Generator

---

**Q35.** `yield`는 어떤 역할을 하는가? / What does `yield` do? / `yield` làm gì?
```python
def count_up(n):
    for i in range(n):
        yield i
```
- A) 함수를 즉시 종료 / Exits the function immediately / Thoát hàm ngay lập tức
- B) 값을 하나씩 돌려주고, 다음 호출 때 이어서 실행 / Returns values one by one, resumes on next call / Trả về từng giá trị, tiếp tục khi được gọi lại
- C) 에러를 발생시킴 / Raises an error / Gây ra lỗi
- D) 리스트를 한번에 반환 / Returns a list at once / Trả về list một lần

---

**Q36.** `yield`를 사용하는 함수를 무엇이라 부르는가? / What is a function with `yield` called? / Hàm có `yield` được gọi là gì?
- A) 데코레이터 / Decorator / Decorator
- B) 제너레이터 / Generator / Generator
- C) 이터레이터 / Iterator / Iterator
- D) 콜백 / Callback / Callback

---

**Q37.** 제너레이터의 가장 큰 장점은? / What is the biggest advantage of generators? / Ưu điểm lớn nhất của generator là gì?
- A) 코드가 짧아진다 / Shorter code / Code ngắn hơn
- B) 데이터를 한번에 메모리에 올리지 않아 메모리 절약 / Saves memory by not loading all data at once / Tiết kiệm bộ nhớ vì không tải toàn bộ dữ liệu cùng lúc
- C) 실행 속도가 10배 빠름 / 10x faster execution / Thực thi nhanh gấp 10 lần
- D) 에러가 발생하지 않음 / No errors occur / Không xảy ra lỗi

---

## Section 9: FastAPI 기초 / FastAPI Basics / FastAPI Cơ bản

---

**Q38.** 다음 코드에서 `@app.get("/hello")`의 의미는? / What does `@app.get("/hello")` mean? / `@app.get("/hello")` có nghĩa là gì?
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello"}
```
- A) `/hello` 경로의 GET 요청을 처리 / Handles GET requests to `/hello` / Xử lý yêu cầu GET đến `/hello`
- B) `/hello` 파일을 읽음 / Reads the `/hello` file / Đọc file `/hello`
- C) hello 함수를 삭제 / Deletes the hello function / Xóa hàm hello
- D) DB에 접속 / Connects to database / Kết nối database

---

**Q39.** `/users/42`로 요청하면 `user_id`의 값은? / What is `user_id` when requesting `/users/42`? / `user_id` có giá trị gì khi gọi `/users/42`?
```python
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}
```
- A) `"42"` (문자열 / string / chuỗi)
- B) `42` (정수 / integer / số nguyên)
- C) `None`
- D) Error / 에러 / Lỗi

---

**Q40.** FastAPI에서 `{"message": "Hello"}`를 반환하면 클라이언트에게 어떤 형태로 전달되는가? / What format does the client receive? / Client nhận được định dạng gì?
- A) HTML
- B) XML
- C) JSON
- D) 일반 텍스트 / Plain text / Văn bản thuần
