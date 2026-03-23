# Quiz Feedback — nguyen
# 퀴즈 피드백 / Quiz Feedback / Phan hoi bai kiem tra

**Date / 날짜 / Ngay**: 2026-03-23

---

## Score / 점수 / Diem

| | Score / 점수 / Diem |
|---|---|
| Part 1: Python Basics (Q1-30) | 25/30 |
| Part 2: Async & FastAPI (Q31-40) | 10/10 |
| **Total / 합계 / Tong** | **35/40 (88%)** |
| **Grade / 등급 / Xep hang** | **B (Good / 양호 / Tot)** |

---

## Your Answers / 제출 답안 / Dap an cua ban

| Q | Your Answer / 답 / Dap an | Correct / 정답 / Dung | Result |
|---|---|---|---|
| 1 | B | B | O |
| **2** | **B** | **C** | **X** |
| 3 | B | B | O |
| 4 | B | B | O |
| **5** | **B** | **C** | **X** |
| 6 | A | A | O |
| **7** | **A** | **B** | **X** |
| 8 | B | B | O |
| 9 | B | B | O |
| 10 | C | C | O |
| 11 | B | B | O |
| **12** | **C** | **B** | **X** |
| 13 | A | A | O |
| **14** | **B** | **A** | **X** |
| 15 | B | B | O |
| 16 | B | B | O |
| 17 | B | B | O |
| 18 | B | B | O |
| 19 | B | B | O |
| 20 | B | B | O |
| 21 | C | C | O |
| 22 | B | B | O |
| 23 | B | B | O |
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

### Q2. 유효하지 않은 변수명은? / Which is NOT a valid variable name? / Ten bien nao KHONG hop le?

- **Your answer / 제출 답**: B) `_count`
- **Correct / 정답**: C) `2nd_value`

**Explanation / 해설 / Giai thich**:

`_count`는 유효한 변수명입니다. 밑줄(`_`)로 시작하는 것은 허용됩니다.
`_count` is a valid variable name. Starting with underscore (`_`) is allowed.
`_count` la ten bien hop le. Bat dau bang dau gach duoi (`_`) duoc cho phep.

`2nd_value`는 숫자로 시작하므로 **불가능**합니다.
`2nd_value` starts with a number, so it is **invalid**.
`2nd_value` bat dau bang so, nen **khong hop le**.

```python
_count = 10      # OK
2nd_value = 20   # SyntaxError!
```

**Rule / 규칙 / Quy tac**: 변수명은 문자(a-z, A-Z) 또는 `_`로 시작해야 합니다. 숫자로 시작할 수 없습니다.
Variable names must start with a letter or `_`. Cannot start with a number.
Ten bien phai bat dau bang chu cai hoac `_`. Khong the bat dau bang so.

---

### Q5. None에 대한 올바른 설명은? / Which is correct about None? / Mo ta dung ve None?

- **Your answer / 제출 답**: B) `None`은 빈 문자열 `""`과 같다 / None equals "" / None bang ""
- **Correct / 정답**: C) `None`은 값이 없음을 나타내는 고유한 타입이다 / None is a unique type meaning "no value" / None la kieu rieng biet nghia la "khong co gia tri"

**Explanation / 해설 / Giai thich**:

`None`은 빈 문자열(`""`)과 **다릅니다**. `None`은 `NoneType`이라는 고유한 타입입니다.
`None` is **not the same** as empty string `""`. `None` has its own type: `NoneType`.
`None` **khong giong** chuoi rong `""`. `None` co kieu rieng: `NoneType`.

```python
print(None == "")     # False
print(None == 0)      # False
print(None == False)  # False
print(type(None))     # <class 'NoneType'>
```

**Key point / 핵심 / Diem chinh**: `None` ≠ `""` ≠ `0` ≠ `False` — 모두 다른 값입니다 / All different values / Tat ca deu khac nhau.

---

### Q7. 문자열을 대문자로 바꾸는 메서드는? / Which method converts to uppercase? / Phuong thuc nao chuyen thanh chu hoa?

- **Your answer / 제출 답**: A) `s.capitalize()`
- **Correct / 정답**: B) `s.upper()`

**Explanation / 해설 / Giai thich**:

```python
s = "hello world"
print(s.capitalize())  # "Hello world"  ← 첫 글자만 / first letter only / chi chu dau
print(s.upper())       # "HELLO WORLD"  ← 전체 대문자 / all uppercase / toan bo chu hoa
print(s.title())       # "Hello World"  ← 각 단어 첫 글자 / first of each word / chu dau moi tu
```

| Method | Result | Description |
|--------|--------|-------------|
| `capitalize()` | `Hello world` | 첫 글자만 대문자 / First letter only / Chi chu dau tien |
| `upper()` | `HELLO WORLD` | 전체 대문자 / All uppercase / Toan bo chu hoa |
| `title()` | `Hello World` | 각 단어 첫 글자 / Each word capitalized / Viet hoa moi tu |

---

### Q12. `[1, 2, 3, 4, 5][1:4]`의 결과는? / What is the result? / Ket qua la gi?

- **Your answer / 제출 답**: C) `[2, 3, 4, 5]`
- **Correct / 정답**: B) `[2, 3, 4]`

**Explanation / 해설 / Giai thich**:

Python 슬라이싱에서 끝 인덱스는 **포함되지 않습니다**.
In Python slicing, the end index is **NOT included**.
Trong Python slicing, chi so cuoi **KHONG duoc bao gom**.

```python
nums = [1, 2, 3, 4, 5]
#       0  1  2  3  4   ← index

nums[1:4]  # index 1, 2, 3 → [2, 3, 4]
#    ^  ^
#    |  └── end (NOT included / 미포함 / khong bao gom)
#    └──── start (included / 포함 / bao gom)
```

**Rule / 규칙 / Quy tac**: `list[start:end]` = index start **이상**, end **미만**
`list[start:end]` = from start (included) to end (excluded)
`list[start:end]` = tu start (bao gom) den end (khong bao gom)

---

### Q14. `print(d["name"])`의 출력은? / What is the output? / Ket qua dau ra la gi?

```python
d = {"name": "Alice", "age": 30}
print(d["name"])
```

- **Your answer / 제출 답**: B) `"Alice"` (따옴표 포함 / with quotes / co dau ngoac kep)
- **Correct / 정답**: A) `Alice` (따옴표 없이 / without quotes / khong co dau ngoac kep)

**Explanation / 해설 / Giai thich**:

`print()` 함수는 문자열의 **내용만** 출력합니다. 따옴표는 표시하지 않습니다.
`print()` outputs only the **content** of a string. No quotes are shown.
`print()` chi xuat **noi dung** cua chuoi. Khong hien thi dau ngoac kep.

```python
name = "Alice"
print(name)   # Alice     ← print() 출력 (따옴표 없음)
repr(name)    # "'Alice'" ← 따옴표 포함된 표현
```

**Key point / 핵심 / Diem chinh**: `print()` = 사람이 읽는 형태 (따옴표 없음) / human-readable (no quotes) / dang doc duoc (khong co ngoac kep)

---

## Summary / 총평 / Tong ket

좋은 성적입니다. 특히 Part 2(async/await, generator, FastAPI)에서 만점을 받아 백엔드 개발에 필요한 고급 개념을 잘 이해하고 있습니다.

틀린 5문제는 모두 Python 기초 영역(변수, 문자열, 컬렉션)의 세부 규칙에 관한 것입니다. 아래 핵심 규칙을 복습하면 빠르게 보완할 수 있습니다:

Good performance. Perfect score on Part 2 (async/await, generators, FastAPI) shows strong understanding of advanced backend concepts.

The 5 wrong answers are all about detailed rules in Python basics (variables, strings, collections). Reviewing these key rules will quickly fill the gaps:

Ket qua tot. Diem tuyet doi o Part 2 (async/await, generator, FastAPI) cho thay hieu biet vung chac ve cac khai niem backend nang cao.

5 cau sai deu lien quan den quy tac chi tiet trong Python co ban (bien, chuoi, collection). On lai cac quy tac chinh sau se nhanh chong bo sung:

### Review Checklist / 복습 체크리스트 / Danh sach on tap

- [ ] 변수명은 숫자로 시작 불가 / Variable names cannot start with numbers / Ten bien khong the bat dau bang so
- [ ] `None` ≠ `""` ≠ `0` ≠ `False`
- [ ] `upper()` = 전체 대문자, `capitalize()` = 첫 글자만 / upper=all, capitalize=first only
- [ ] `list[start:end]` — end는 미포함 / end is excluded / end khong bao gom
- [ ] `print()` — 따옴표 없이 출력 / prints without quotes / in ra khong co ngoac kep
