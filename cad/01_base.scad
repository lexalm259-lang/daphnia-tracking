petri_diameter = 80;  // Диаметр чашки Петри


box_width = 130;      // Внутренняя ширина коробки
box_height = 60;      // Внутренняя высота коробки
wall_thickness = 5;   // Толшина стенки
bottom_thickness = 3; // Толнина дна

slot_height = 5;            // Ширина паза для установки трубы 
slot_width = slot_height/2; // Толшина паза для установки трубы

doorslot_width = 5;       // Толшина паза для установки шторки
doorslot_depth = 2.1;     // Ширина паза для установки шторки
doorslot_thickness = 4;   // Толшина выступа для монтажа шторки
door_width = 100;         // ширина отверстия двери
door_height = box_height; // высота отверстия двери
eps = 0.02;


module lightbox_base() {
    difference() {
        cube([
            box_width + 2*wall_thickness,
            box_width + 2*wall_thickness,
            box_height + bottom_thickness
        ]);
        translate([wall_thickness, wall_thickness, bottom_thickness]) {
            cube([box_width, box_width, box_height]);
        }
        translate([// Оверстие двери
            wall_thickness + (box_width - door_width)/2,
            -1,
            bottom_thickness + (box_height - door_height)/2
        ]) {
            cube([door_width, wall_thickness + 2, door_height]);
        }
        translate([ //Паз для установки верхней трубы
            wall_thickness-slot_width,
            wall_thickness-slot_width, 
            box_height+bottom_thickness - slot_height
        ]) {
            cube([box_width+2*slot_width, box_width+2*slot_width, slot_height]);
        }
    };
};
module doorslot() {//Паз для установки двери
    difference() {
        cube([doorslot_thickness, door_width+2*doorslot_width, door_height+bottom_thickness]);
        translate([doorslot_depth, doorslot_width-doorslot_depth, bottom_thickness]) {
            cube([doorslot_depth, door_width+2*doorslot_depth, door_height]);
        }
        translate([0, doorslot_width, bottom_thickness]) {
            cube([doorslot_depth, door_width, door_height]);
        }
    }
        
};
 
module Box() {//Паз для установки двери
    lightbox_base();
    translate([
        wall_thickness + (box_width - door_width)/2 - doorslot_width,
        -doorslot_thickness,  
        0
    ]){
        mirror ([0,1,0]) rotate([0,0,-90]) doorslot();
    }
};

Box();

