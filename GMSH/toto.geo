// This code was created by pyPLANES
Point(1)= {0, 0, 0.0,0.03};
Point(2)= {0.1, 0, 0.0,0.03};
Point(3)= {0.1, 0.1, 0.0,0.03};
Point(4)= {0, 0.1, 0.0,0.03};
Line(5)= {1, 2};
Line(6)= {2, 3};
Line(7)= {3, 4};
Line(8)= {4, 1};
Line Loop(9)= {5,6 ,7 ,8 };
Plane Surface(10)= {9};
Physical Line("condition=Transmission")={7};
Physical Line("condition=Periodicity")={6, 8};
Physical Line("condition=Incident_PW")={5};
Physical Surface("mat=foam2 98 1")={10};
Physical Line("model=FEM1D")={5, 6, 8, 7};
Physical Surface("model=FEM2D")={10};
Periodic Line {6} = {8} Translate {0.1,0,0};
