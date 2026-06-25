petri_diameter = 80;  // Диаметр чашки Петри


box_width = 100;      // Внутренняя ширина коробки
box_height = 60;      // Внутренняя высота коробки
wall_thickness = 5;   // Толшина стенки
bottom_thickness = 5; // Толнина дна

cap_width = 100;

border_width = box_width-cap_width;      // Ширина Кольца освещения в проекции на землю
//border_height = 100-80+20-5;     // высота кольца
border_height = 40;     // высота кольца

slot_height = 5;            // Ширина паза для установки трубы 
slot_width = slot_height/2; // Толшина паза для установки трубы

doorslot_width = 5;       // Толшина паза для установки шторки
doorslot_depth = 2.1;     // Ширина паза для установки шторки
doorslot_thickness = 4;   // Толшина выступа для монтажа шторки
door_width = 100;         // ширина отверстия двери
door_height = box_height; // высота отверстия двери
eps = 0.02;


// Модуль пустотелой усеченной пирамиды
module lighting_ring() {
    // Проверка на корректность размеров (комментарий)
    // assert(a_bot > 2*t && a_top > 2*t);
    bot_out_width = box_width+2*wall_thickness;
    bot_in_width = box_width;
    top_out_width = bot_out_width-border_width;
    top_in_width = bot_in_width-border_width;
    
    // Верхний коннектор
    translate([0,0,border_height+slot_height*3/2])
    difference(){
        cube([top_out_width, top_out_width, slot_height],center=true);
        cube([top_in_width+2*slot_width, top_in_width+2*slot_width, slot_height+eps],center=true);
    }
    // Нижний коннектор
    translate([0,0,slot_height/2])
    difference(){
    cube([bot_in_width+2*slot_width, bot_in_width+2*slot_width, slot_height],center=true);
    cube([bot_in_width, bot_in_width, slot_height+eps],center=true);
    }
    translate([0,0,slot_height])
    difference() {
        // Внешняя усеченная пирамида
        linear_extrude(height=border_height, center=false, scale=top_out_width/bot_out_width)
            square([bot_out_width, bot_out_width], center=true);
        // Внутренняя усеченная пирамида
        translate([0,0,-eps])
        linear_extrude(height=border_height+2*eps, center=false, scale=top_in_width/ bot_in_width)
                square([bot_in_width, bot_in_width], center=true);
    }
}
lighting_ring();
