petri_diameter = 80;  // Диаметр чашки Петри

box_width = 100;      // Внутренняя ширина коробки
box_height = 60;      // Внутренняя высота коробки
wall_thickness = 5;   // Толщина стенки
bottom_thickness = 5; // Толщина дна

cap_width = 100;
border_width = box_width - cap_width;
border_height = 10 - 5;

slot_height = 5;
slot_width = slot_height / 2;

doorslot_width = 5;
doorslot_depth = 2.1;
doorslot_thickness = 4;
door_width = 100;
door_height = box_height;
eps = 0.02;

// --- Параметры отверстия в углу ---
hole_size = 15;              // Размер отверстия (мм)
hole_dist_from_top = 15;     // Отступ от верха трубы до центра отверстия (мм)

// --- Параметры ПРЯМОУГОЛЬНОГО отверстия внизу стенки ---
lower_hole_width = 90;       // Ширина отверстия (по X)
lower_hole_height = 25;      // Высота отверстия (по Z)
lower_hole_from_bottom = 0;  // Отступ от дна до нижнего края отверстия
lower_hole_center_x = 0;     // Смещение центра отверстия по X (0 = по центру)

// Параметры трубы
outer_size = 100;
aperture_height = 90;
aperture_thickness = 0.8;    
    
inner_size = outer_size - 2 * aperture_thickness;

// === МОДУЛЬ КОЛЬЦА ===
module lighting_ring() {
    bot_out_width = box_width + 2 * wall_thickness;
    bot_in_width = box_width;
    top_out_width = bot_out_width - border_width;
    top_in_width = bot_in_width - border_width;
    
    translate([0, 0, border_height + slot_height * 3 / 2])
    difference() {
        cube([top_out_width, top_out_width, slot_height], center = true);
        cube([top_in_width + 2 * slot_width, top_in_width + 2 * slot_width, slot_height + eps], center = true);
    }
    translate([0, 0, slot_height / 2])
    difference() {
        cube([bot_in_width + 2 * slot_width, bot_in_width + 2 * slot_width, slot_height], center = true);
        cube([bot_in_width, bot_in_width, slot_height + eps], center = true);
    }
    translate([0, 0, slot_height])
    difference() {
        linear_extrude(height = border_height, center = false, scale = top_out_width / bot_out_width)
            square([bot_out_width, bot_out_width], center = true);
        translate([0, 0, -eps])
        linear_extrude(height = border_height + 2 * eps, center = false, scale = top_in_width / bot_in_width)
            square([bot_in_width, bot_in_width], center = true);
    }
}

// === СБОРКА ===
translate([0, 0, -aperture_height / 2 + 10])
difference() {
    // Внешний корпус трубы
    cube([outer_size, outer_size, aperture_height], center = true);
    
    // Внутренняя полость
    cube([inner_size, inner_size, aperture_height + 20], center = true);
    
    // 🔌 ОТВЕРСТИЕ В УГЛУ (передний правый угол)
    translate([
        outer_size / 2 - hole_size / 2, 
        outer_size / 2 - hole_size / 2,
        aperture_height / 2 - hole_dist_from_top
    ])
    cube([hole_size, hole_size, hole_size], center = true);
    
    // 🔻 ПРЯМОУГОЛЬНОЕ ОТВЕРСТИЕ НА ЛЕВОЙ СТЕНКЕ
    translate([
        -outer_size / 2,                                          // X: левая грань
        lower_hole_center_x,                                      // Y: смещение вдоль стенки (0 = по центру)
        -aperture_height / 2 + lower_hole_from_bottom + lower_hole_height / 2  // Z: центр по вертикали
    ])
    cube([
        aperture_thickness + 2,               // Толщина (с запасом) по X
        lower_hole_width,                     // Ширина по Y
        lower_hole_height                     // Высота по Z
    ], center = true);
}

// Устанавливаем кольцо сверху
lighting_ring();