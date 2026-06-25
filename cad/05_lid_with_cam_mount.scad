$fn = 50;

cap_width = 100;
cap_height = 10;
D_mount = 6.5;       // Диаметр монтировки
D_inner_mount = 2.5; // Диаметр отверстия под стержень (D_mount - 2*thickness)
width_mount = 32;    // Ширина монтировки
dist_mount = 8;      // Ширина пропила под шип камеры
shift_mount = 12;    // Смещение объектива относительно центра камеры
shift_distance = 10; // Смещение объектива вверх относительно плоскости

wall_thickness = 5;  // Толщина стенки

slot_height = 5;            // Ширина паза
slot_width = slot_height/2; // Толщина паза

D_hole = 10; // Диаметр отверстия для проводов

EPS = 0.01; 

notch_depth = 0.5;
notch_width = 2;
notch_height = 2;
notch_offset = 2;

// ============================================================
// МОДУЛЬ КРЕПЛЕНИЯ КАМЕРЫ (ОБНОВЛЁННАЯ ВЕРСИЯ)
// ============================================================
module cam_mount(width, D, D_inner, space) {
     difference() {
        union() {
            translate([0, 0, D/4+shift_distance/2]) 
                cube([D, width, D/2+shift_distance], center = true); // Прямоугольная подпорка
            
            translate([0, 0, D/2+shift_distance]) rotate([90,0,0]) 
                cylinder(h = width, d = D, center = true); // Цилиндрическая подпорка
            
            // 🔥 ОБНОВЛЕНИЕ: скруглённое основание крепления камеры
            linear_extrude(height = 2, center = true)
                offset(r = 5)
                square([30 - 2*5, 50 - 2*5], center = true);
        };
        
        translate([0, 0, D/2+shift_distance]) rotate([90,0,00]) 
            cylinder(h = width + 1, d = D_inner, center = true); // Отверстие для винта
        
        translate([0, 0, D/2+shift_distance]) 
            cube([D+EPS, space, D+0.], center = true); // Центральный вырез
     };
    
    // Упор
    translate([0, width_mount/2 - notch_offset, D_mount/2 - notch_height])
        cube([notch_width, notch_depth, notch_height/2], center = true);
}

// ============================================================
// МОДУЛЬ КРЫШКИ
// ============================================================
module cap() {
    width = cap_width + 2*slot_width;
    out_width = cap_width + 2*wall_thickness;
    out_height = cap_height - slot_height;

    cube([width, width, cap_height], center = true);  // крышка 110×110×10
    translate([0, 0, -cap_height/2 + slot_height/2]) 
        cube([out_width, out_width, out_height], center = true);
}

// ============================================================
// МОДУЛЬ ОТВЕРСТИЯ ДЛЯ ПРОВОДОВ
// ============================================================
module hole(length, width) {
    h = 100;
    cube_length = length - width;
    cube([cube_length + EPS, width, h], center = true);
    translate([cube_length/2, 0, 0]) 
        cylinder(h = h, d = width, center = true);
}

// ============================================================
// СБОРКА: крышка + крепление камеры - отверстие для проводов
// ============================================================
difference() {
    union() {
        cap();   // Крышка
        rotate([0, 0, 45]) translate([0, shift_mount, cap_height/2])  
            cam_mount(width_mount, D_mount, D_inner_mount, dist_mount); // Крепление для камеры
    }
    // Вырез для провода
    translate([-cap_width/2, 0, 0]) hole(2*D_hole, D_hole);
}