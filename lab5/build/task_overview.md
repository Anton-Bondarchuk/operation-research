

**SLSQP (Sequential Least Squares Programming)** - алгоритм последовательного квадратичного программирования.

SLSQP минимизирует функцию нескольких переменных с любой комбинацией ограничений границ, равенства и неравенства. Метод оборачивает подпрограмму оптимизации SLSQP, изначально реализованную Дитером Крафтом.

достаточно быстрый тк 


Метод SQP: Основная идея метода заключается в решении серии квадратичных программ, где каждая следующая подзадача является приближением второй производной функции с использованием Лагранжевых множителей для учёта ограничений.

Использование BFGS: Алгоритм использует метод БФГС (Broyden–Fletcher–Goldfarb–Shanno) для обновления матрицы Гессе, что помогает ускорить сходимость.

Ограничения: Он может работать с любыми комбинациями ограничений — равенствами, неравенствами и ограничениями на переменные.

