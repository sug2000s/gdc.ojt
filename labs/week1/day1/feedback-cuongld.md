# Quiz Feedback — cuongld
# 퀴즈 피드백 / Quiz Feedback / Phan hoi bai kiem tra

**Date / 날짜 / Ngay**: 2026-03-23

---

## Score / 점수 / Diem

| | Score / 점수 / Diem |
|---|---|
| Part 1: Python Basics (Q1-30) | 28/30 |
| Part 2: Async & FastAPI (Q31-40) | 10/10 |
| **Total / 합계 / Tong** | **38/40 (95%)** |
| **Grade / 등급 / Xep hang** | **A (Excellent / 우수 / Xuat sac)** |

---

## Your Answers / 제출 답안 / Dap an cua ban

| Q | Your Answer / 답 / Dap an | Correct / 정답 / Dung | Result |
|---|---|---|---|
| 1 | B | B | O |
| 2 | C | C | O |
| 3 | B | B | O |
| 4 | B | B | O |
| 5 | C | C | O |
| 6 | A | A | O |
| 7 | B | B | O |
| 8 | B | B | O |
| 9 | B | B | O |
| 10 | C | C | O |
| 11 | B | B | O |
| 12 | B | B | O |
| 13 | A | A | O |
| 14 | A | A | O |
| 15 | B | B | O |
| 16 | B | B | O |
| 17 | B | B | O |
| 18 | B | B | O |
| 19 | B | B | O |
| 20 | B | B | O |
| 21 | C | C | O |
| **22** | **A** | **B** | **X** |
| **23** | **A** | **B** | **X** |
| 24 | B | B | O |
| 25 | A | A | O |
| 26 | C | C | O |
| 27 | B | B | O |
| 28 | B | B | O |
| 29 | C | C | O |
| 30 | B | B | O |
| 31 | B | B | O |
| 32 | B | B | O |
| 33 | C | C | O |
| 34 | B | B | O |
| 35 | B | B | O |
| 36 | B | B | O |
| 37 | B | B | O |
| 38 | A | A | O |
| 39 | B | B | O |
| 40 | C | C | O |

---

## Wrong Answers Explained / 오답 해설 / Giai thich cau sai

### Q22. `*args`의 역할은? / What does `*args` do? / `*args` lam gi?

```python
def func(*args):
    print(args)
```

- **Your answer / 제출 답**: A) 키워드 인자를 딕셔너리로 받음 / Receives keyword args as dict / Nhan doi so keyword dang dict
- **Correct / 정답**: B) 위치 인자를 튜플로 받음 / Receives positional args as tuple / Nhan doi so vi tri dang tuple

**Explanation / 해설 / Giai thich**:

`*args`는 별(star) 1개 = 위치 인자(positional arguments)를 **tuple**로 받습니다.
`*args` with one star = receives positional arguments as a **tuple**.
`*args` voi mot dau sao = nhan doi so vi tri dang **tuple**.

```python
def example(*args):
    print(type(args))  # <class 'tuple'>
    print(args)

example(1, 2, 3)  # (1, 2, 3)
```

---

### Q23. `**kwargs`의 역할은? / What does `**kwargs` do? / `**kwargs` lam gi?

- **Your answer / 제출 답**: A) 위치 인자를 튜플로 받음 / Receives positional args as tuple / Nhan doi so vi tri dang tuple
- **Correct / 정답**: B) 키워드 인자를 딕셔너리로 받음 / Receives keyword args as dict / Nhan doi so keyword dang dict

**Explanation / 해설 / Giai thich**:

`**kwargs`는 별(star) 2개 = 키워드 인자(keyword arguments)를 **dict**로 받습니다.
`**kwargs` with two stars = receives keyword arguments as a **dict**.
`**kwargs` voi hai dau sao = nhan doi so keyword dang **dict**.

```python
def example(**kwargs):
    print(type(kwargs))  # <class 'dict'>
    print(kwargs)

example(name="Alice", age=30)  # {'name': 'Alice', 'age': 30}
```

**Quick tip / 외우는 법 / Meo nho**:
- `*` (1 star) → simple → **tuple** (순서만 있음 / ordered only / chi co thu tu)
- `**` (2 stars) → key-value → **dict** (이름=값 / name=value / ten=gia tri)

---

## Summary / 총평 / Tong ket

우수한 성적입니다. 40문제 중 38문제 정답으로, Python 기초와 async/FastAPI 모두 높은 이해도를 보여줍니다. 틀린 2문제는 `*args`와 `**kwargs`의 역할을 서로 바꿔 답한 것으로, 개념 자체는 알고 있으나 매핑만 혼동한 것입니다.

Excellent performance. 38 out of 40 correct, showing strong understanding of both Python basics and async/FastAPI. The 2 wrong answers were simply swapping `*args` and `**kwargs` roles — the concepts are understood, just the mapping was reversed.

Ket qua xuat sac. Dung 38/40 cau, the hien su hieu biet vung chac ve Python co ban va async/FastAPI. 2 cau sai chi la do nham lan vai tro cua `*args` va `**kwargs` — khai niem da nam duoc, chi can nho chinh xac hon.
