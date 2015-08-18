model Circuit
    Real u0, u1, u2, u3, uL;
    Real i0, i1, i2, i3, iL(start=0);
    parameter Real R1 = 1;
    parameter Real R2 = 1;
    parameter Real R3 = 1;
    parameter Real L = 1;
    parameter Real C = 1;
    parameter Real omega = 1;
equation
    u0 = sin(omega*time);
    u1 = R1 * i1;
    u2 = R2 * i2;
    
    u3 = R3 * i3; // Resistor
    // i3 = C * der(u3); // Capacitor
    
    uL = L * der(iL);
    u0 = u1 + u3;
    uL = u1 + u2;
    u2 = u3;
    i0 = i1 + iL;
    i1 = i2 + i3;
end Circuit;
