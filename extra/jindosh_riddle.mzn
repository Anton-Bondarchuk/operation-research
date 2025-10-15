include "all_different.mzn";

enum Lady = {Winslow, Marcolla, Conti, Natsiou, Finch};

set of int: Position = 1..5;

enum Color = {Blue, White, Red, Purple, Green};

enum City = {Morley, Freyport, Dunwall, Serkanos, Baleton};

enum Drink = {Whiskey, Absinthe, Cider, Rum, Beer};

enum Jewel = {Cigarette_Case, Bird_Pendant, Ring, Diamond, Medal};

array[Lady] of var Position: lady_pos;

array[Position] of var Color: color_at_pos;
array[Position] of var City: city_at_pos;
array[Position] of var Drink: drink_at_pos;
array[Position] of var Jewel: jewel_at_pos;

% все позиции разные 
% Гарантирует, что все дамы сидят в разных позициях, и все характеристики уникальны
constraint all_different(lady_pos);

constraint all_different(color_at_pos);
constraint all_different(city_at_pos);
constraint all_different(drink_at_pos);
constraint all_different(jewel_at_pos);


% Леди Уинслоу носила синее пончо
constraint color_at_pos[lady_pos[Winslow]] = Blue;

% Доктор Марколла сидела левее всех (позиция 1)
constraint lady_pos[Marcolla] = 1;

% Доктор Марколла рядом с гостьей в белом платье
constraint color_at_pos[2] = White;

% Дама в красном платье сидела слева от дамы в пурпурном
constraint exists(p in 1..4)(
  color_at_pos[p] = Red /\ color_at_pos[p+1] = Purple
);

% Обладательница красного платья пролила виски
constraint forall(p in Position)(
  color_at_pos[p] = Red -> drink_at_pos[p] = Whiskey
);

% Путешественница из Морли была одета в зеленое
constraint forall(p in Position)(
  city_at_pos[p] = Morley -> color_at_pos[p] = Green
);

% Женщина рядом с обладательницей Портсигара - из Морли
constraint exists(p in 1..4)(
  (jewel_at_pos[p] = Cigarette_Case /\ city_at_pos[p+1] = Morley) \/
  (jewel_at_pos[p+1] = Cigarette_Case /\ city_at_pos[p] = Morley)
);

% Баронесса Финч имеет Кулон с птицей
constraint jewel_at_pos[lady_pos[Finch]] = Bird_Pendant;

% Дама из Фрейпорта имеет Перстень
constraint forall(p in Position)(
  city_at_pos[p] = Freyport -> jewel_at_pos[p] = Ring
);

% Женщина рядом с обладательницей Бриллианта - из Дануолла
constraint exists(p in 1..4)(
  (jewel_at_pos[p] = Diamond /\ city_at_pos[p+1] = Dunwall) \/
  (jewel_at_pos[p+1] = Diamond /\ city_at_pos[p] = Dunwall)
);

% Графиня Конти пила абсент
constraint drink_at_pos[lady_pos[Conti]] = Absinthe;

% Дама, направляющаяся в Серконос, пила сидр
constraint forall(p in Position)(
  city_at_pos[p] = Serkanos -> drink_at_pos[p] = Cider
);

% Дама посередине пролила ром
constraint drink_at_pos[3] = Rum;

% Мадам Нациу из Бейлтона
constraint city_at_pos[lady_pos[Natsiou]] = Baleton;

solve satisfy;

output [
  "Jindosh Riddle Solution:\n",
  "Positions: ", show(lady_pos), "\n",
  "Colors: ", show(color_at_pos), "\n",
  "Cities: ", show(city_at_pos), "\n", 
  "Drinks: ", show(drink_at_pos), "\n",
  "Jewels: ", show(jewel_at_pos), "\n\n",
  "Winslow at position: ", show(lady_pos[Winslow]), "\n",
  "Marcolla at position: ", show(lady_pos[Marcolla]), "\n",
  "Conti at position: ", show(lady_pos[Conti]), "\n",
  "Natsiou at position: ", show(lady_pos[Natsiou]), "\n",
  "Finch at position: ", show(lady_pos[Finch]), "\n",
];