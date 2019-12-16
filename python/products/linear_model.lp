\* Source Pyomo model name=unknown *\

min 
objective:
-0.01 pi(17)
+0.01 pi(18)
-0.02 pi(8)
+0.02 pi(9)
+3000 psi(10)
+3000 psi(12)
+3000 psi(15)
+3000 psi(16)
+3000 psi(19)
+3000 psi(20)
+3000 psi(3)
+3000 psi(6)
+3000 psi(7)
+138894 ONE_VAR_CONSTANT

s.t.

c_e_gas_flow_linear(1)_:
+10.911288000000001 phi(1)
-9.0702700000000007 pi(1)
+9.0702700000000007 pi(2)
= 0

c_e_gas_flow_linear(2)_:
+10.911288000000001 phi(2)
-9.0702700000000007 pi(1)
+9.0702700000000007 pi(2)
= 0

c_e_gas_flow_linear(3)_:
+19.311288000000001 phi(3)
-6.0468500000000001 pi(2)
+6.0468500000000001 pi(3)
= 0

c_e_gas_flow_linear(4)_:
+19.311288000000001 phi(4)
-6.0468500000000001 pi(2)
+6.0468500000000001 pi(3)
= 0

c_e_gas_flow_linear(5)_:
+30.786576 phi(5)
-1.3954299999999999 pi(3)
+1.3954299999999999 pi(4)
= 0

c_e_gas_flow_linear(6)_:
+5.6294240000000002 phi(6)
-0.100256 pi(5)
+0.100256 pi(6)
= 0

c_e_gas_flow_linear(7)_:
+2.4385759999999999 phi(7)
-0.14865500000000001 pi(6)
+0.14865500000000001 pi(7)
= 0

c_e_gas_flow_linear(8)_:
+12.950576 phi(8)
+0.22689500000000001 pi(4)
-0.22689500000000001 pi(7)
= 0

c_e_gas_flow_linear(9)_:
+17.835999999999999 phi(9)
+0.65965600000000002 pi(14)
-0.65965600000000002 pi(4)
= 0

c_e_gas_flow_linear(12)_:
+39.236448000000003 phi(12)
+1.8140499999999999 pi(10)
-1.8140499999999999 pi(9)
= 0

c_e_gas_flow_linear(13)_:
+4.7875519999999998 phi(13)
+0.027008399999999998 pi(10)
-0.027008399999999998 pi(9)
= 0

c_e_gas_flow_linear(14)_:
+27.890817999999999 phi(14)
-1.4512400000000001 pi(10)
+1.4512400000000001 pi(11)
= 0

c_e_gas_flow_linear(15)_:
+3.4031820000000002 phi(15)
-0.0216067 pi(10)
+0.0216067 pi(11)
= 0

c_e_gas_flow_linear(16)_:
+27.012 phi(16)
-0.86383600000000005 pi(11)
+0.86383600000000005 pi(12)
= 0

c_e_gas_flow_linear(17)_:
+22.771999999999998 phi(17)
-0.90702700000000003 pi(12)
+0.90702700000000003 pi(13)
= 0

c_e_gas_flow_linear(18)_:
+25.172000000000001 phi(18)
-7.2562199999999999 pi(13)
+7.2562199999999999 pi(14)
= 0

c_e_gas_flow_linear(19)_:
+44.927999999999997 phi(19)
-3.6281099999999999 pi(14)
+3.6281099999999999 pi(15)
= 0

c_e_gas_flow_linear(20)_:
+31.231999999999999 phi(20)
-1.4512400000000001 pi(15)
+1.4512400000000001 pi(16)
= 0

c_e_gas_flow_linear(21)_:
+4.282 phi(21)
-0.051444499999999997 pi(11)
+0.051444499999999997 pi(17)
= 0

c_e_gas_flow_linear(23)_:
+4.282 phi(23)
-0.0017032 pi(18)
+0.0017032 pi(19)
= 0

c_e_gas_flow_linear(24)_:
+3.8380000000000001 phi(24)
-0.027819 pi(19)
+0.027819 pi(20)
= 0

c_l_operational_1(10)_:
-1 pi(8)
+1 pi(9)
>= 0

c_l_operational_1(11)_:
-1 pi(8)
+1 pi(9)
>= 0

c_l_operational_1(22)_:
-1 pi(17)
+1 pi(18)
>= 0

c_u_operational_2(10)_:
-4 pi(8)
+1 pi(9)
<= 0

c_u_operational_2(11)_:
-4 pi(8)
+1 pi(9)
<= 0

c_u_operational_2(22)_:
-4 pi(17)
+1 pi(18)
<= 0

c_l_operational_3(10)_:
+1 phi(10)
>= 0

c_l_operational_3(11)_:
+1 phi(11)
>= 0

c_l_operational_3(22)_:
+1 phi(22)
>= 0

c_l_security_1(1)_:
+1 pi(1)
>= 0

c_l_security_1(2)_:
+1 pi(2)
>= 0

c_l_security_1(3)_:
+1 pi(3)
>= 900

c_l_security_1(4)_:
+1 pi(4)
>= 0

c_l_security_1(5)_:
+1 pi(5)
>= 0

c_l_security_1(6)_:
+1 pi(6)
>= 900

c_l_security_1(7)_:
+1 pi(7)
>= 900

c_l_security_1(8)_:
+1 pi(8)
>= 2500

c_l_security_1(9)_:
+1 pi(9)
>= 0

c_l_security_1(10)_:
+1 pi(10)
>= 900

c_l_security_1(11)_:
+1 pi(11)
>= 0

c_l_security_1(12)_:
+1 pi(12)
>= 0

c_l_security_1(13)_:
+1 pi(13)
>= 0

c_l_security_1(14)_:
+1 pi(14)
>= 0

c_l_security_1(15)_:
+1 pi(15)
>= 0

c_l_security_1(16)_:
+1 pi(16)
>= 2500

c_l_security_1(17)_:
+1 pi(17)
>= 0

c_l_security_1(18)_:
+1 pi(18)
>= 0

c_l_security_1(19)_:
+1 pi(19)
>= 0

c_l_security_1(20)_:
+1 pi(20)
>= 625

c_u_security_2(1)_:
+1 pi(1)
<= 5929

c_u_security_2(2)_:
+1 pi(2)
<= 5929

c_u_security_2(3)_:
+1 pi(3)
<= 6400

c_u_security_2(4)_:
+1 pi(4)
<= 6400

c_u_security_2(5)_:
+1 pi(5)
<= 5929

c_u_security_2(6)_:
+1 pi(6)
<= 6400

c_u_security_2(7)_:
+1 pi(7)
<= 6400

c_u_security_2(8)_:
+1 pi(8)
<= 4382.4400000000005

c_u_security_2(9)_:
+1 pi(9)
<= 4382.4400000000005

c_u_security_2(10)_:
+1 pi(10)
<= 4382.4400000000005

c_u_security_2(11)_:
+1 pi(11)
<= 4382.4400000000005

c_u_security_2(12)_:
+1 pi(12)
<= 4382.4400000000005

c_u_security_2(13)_:
+1 pi(13)
<= 4382.4400000000005

c_u_security_2(14)_:
+1 pi(14)
<= 4382.4400000000005

c_u_security_2(15)_:
+1 pi(15)
<= 4382.4400000000005

c_u_security_2(16)_:
+1 pi(16)
<= 4382.4400000000005

c_u_security_2(17)_:
+1 pi(17)
<= 4382.4400000000005

c_u_security_2(18)_:
+1 pi(18)
<= 3969

c_u_security_2(19)_:
+1 pi(19)
<= 4382.4400000000005

c_u_security_2(20)_:
+1 pi(20)
<= 4382.4400000000005

c_e_injection(1)_:
-1 phi(1)
-1 phi(2)
+1 psi(1)
= 0

c_e_injection(2)_:
+1 phi(1)
+1 phi(2)
-1 phi(3)
-1 phi(4)
+1 psi(2)
= 0

c_e_injection(3)_:
+1 phi(3)
+1 phi(4)
-1 phi(5)
+1 psi(3)
= 0

c_e_injection(4)_:
+1 phi(5)
+1 phi(8)
-1 phi(9)
+1 psi(4)
= 0

c_e_injection(5)_:
-1 phi(6)
+1 psi(5)
= 0

c_e_injection(6)_:
+1 phi(6)
-1 phi(7)
+1 psi(6)
= 0

c_e_injection(7)_:
+1 phi(7)
-1 phi(8)
+1 psi(7)
= 0

c_e_injection(8)_:
-1 phi(10)
-1 phi(11)
+1 psi(8)
= 0

c_e_injection(9)_:
+1 phi(10)
+1 phi(11)
-1 phi(12)
-1 phi(13)
+1 psi(9)
= 0

c_e_injection(10)_:
+1 phi(12)
+1 phi(13)
-1 phi(14)
-1 phi(15)
+1 psi(10)
= 0

c_e_injection(11)_:
+1 phi(14)
+1 phi(15)
-1 phi(16)
-1 phi(21)
+1 psi(11)
= 0

c_e_injection(12)_:
+1 phi(16)
-1 phi(17)
+1 psi(12)
= 0

c_e_injection(13)_:
+1 phi(17)
-1 phi(18)
+1 psi(13)
= 0

c_e_injection(14)_:
+1 phi(18)
-1 phi(19)
+1 phi(9)
+1 psi(14)
= 0

c_e_injection(15)_:
+1 phi(19)
-1 phi(20)
+1 psi(15)
= 0

c_e_injection(16)_:
+1 phi(20)
+1 psi(16)
= 0

c_e_injection(17)_:
+1 phi(21)
-1 phi(22)
+1 psi(17)
= 0

c_e_injection(18)_:
+1 phi(22)
-1 phi(23)
+1 psi(18)
= 0

c_e_injection(19)_:
+1 phi(23)
-1 phi(24)
+1 psi(19)
= 0

c_e_injection(20)_:
+1 phi(24)
+1 psi(20)
= 0

c_l_contractual_1(1)_:
+1 psi(1)
>= 8.8699999999999992

c_l_contractual_1(2)_:
+1 psi(2)
>= 0

c_l_contractual_1(3)_:
+1 psi(3)
>= -3.9180000000000001

c_l_contractual_1(4)_:
+1 psi(4)
>= 0

c_l_contractual_1(5)_:
+1 psi(5)
>= 0

c_l_contractual_1(6)_:
+1 psi(6)
>= -4.0339999999999998

c_l_contractual_1(7)_:
+1 psi(7)
>= -5.2560000000000002

c_l_contractual_1(8)_:
+1 psi(8)
>= 20.344000000000001

c_l_contractual_1(9)_:
+1 psi(9)
>= 0

c_l_contractual_1(10)_:
+1 psi(10)
>= -6.3650000000000002

c_l_contractual_1(11)_:
+1 psi(11)
>= 0

c_l_contractual_1(12)_:
+1 psi(12)
>= -2.1200000000000001

c_l_contractual_1(13)_:
+1 psi(13)
>= 0

c_l_contractual_1(14)_:
+1 psi(14)
>= 0

c_l_contractual_1(15)_:
+1 psi(15)
>= -6.8479999999999999

c_l_contractual_1(16)_:
+1 psi(16)
>= -15.616

c_l_contractual_1(17)_:
+1 psi(17)
>= 0

c_l_contractual_1(18)_:
+1 psi(18)
>= 0

c_l_contractual_1(19)_:
+1 psi(19)
>= -0.222

c_l_contractual_1(20)_:
+1 psi(20)
>= -1.919

c_u_contractual_2(1)_:
+1 psi(1)
<= 11.593999999999999

c_u_contractual_2(2)_:
+1 psi(2)
<= 8.4000000000000004

c_u_contractual_2(3)_:
+1 psi(3)
<= 0

c_u_contractual_2(4)_:
+1 psi(4)
<= 0

c_u_contractual_2(5)_:
+1 psi(5)
<= 4.7999999999999998

c_u_contractual_2(6)_:
+1 psi(6)
<= 0

c_u_contractual_2(7)_:
+1 psi(7)
<= 0

c_u_contractual_2(8)_:
+1 psi(8)
<= 22.012

c_u_contractual_2(9)_:
+1 psi(9)
<= 0

c_u_contractual_2(10)_:
+1 psi(10)
<= 0

c_u_contractual_2(11)_:
+1 psi(11)
<= 0

c_u_contractual_2(12)_:
+1 psi(12)
<= 0

c_u_contractual_2(13)_:
+1 psi(13)
<= 1.2

c_u_contractual_2(14)_:
+1 psi(14)
<= 0.95999999999999996

c_u_contractual_2(15)_:
+1 psi(15)
<= 0

c_u_contractual_2(16)_:
+1 psi(16)
<= 0

c_u_contractual_2(17)_:
+1 psi(17)
<= 0

c_u_contractual_2(18)_:
+1 psi(18)
<= 0

c_u_contractual_2(19)_:
+1 psi(19)
<= 0

c_u_contractual_2(20)_:
+1 psi(20)
<= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= pi(1) <= +inf
   0 <= pi(2) <= +inf
   0 <= pi(3) <= +inf
   0 <= pi(4) <= +inf
   0 <= pi(5) <= +inf
   0 <= pi(6) <= +inf
   0 <= pi(7) <= +inf
   0 <= pi(8) <= +inf
   0 <= pi(9) <= +inf
   0 <= pi(10) <= +inf
   0 <= pi(11) <= +inf
   0 <= pi(12) <= +inf
   0 <= pi(13) <= +inf
   0 <= pi(14) <= +inf
   0 <= pi(15) <= +inf
   0 <= pi(16) <= +inf
   0 <= pi(17) <= +inf
   0 <= pi(18) <= +inf
   0 <= pi(19) <= +inf
   0 <= pi(20) <= +inf
    -inf <= psi(1) <= +inf
    -inf <= psi(2) <= +inf
    -inf <= psi(3) <= +inf
    -inf <= psi(4) <= +inf
    -inf <= psi(5) <= +inf
    -inf <= psi(6) <= +inf
    -inf <= psi(7) <= +inf
    -inf <= psi(8) <= +inf
    -inf <= psi(9) <= +inf
    -inf <= psi(10) <= +inf
    -inf <= psi(11) <= +inf
    -inf <= psi(12) <= +inf
    -inf <= psi(13) <= +inf
    -inf <= psi(14) <= +inf
    -inf <= psi(15) <= +inf
    -inf <= psi(16) <= +inf
    -inf <= psi(17) <= +inf
    -inf <= psi(18) <= +inf
    -inf <= psi(19) <= +inf
    -inf <= psi(20) <= +inf
    -inf <= phi(1) <= +inf
    -inf <= phi(2) <= +inf
    -inf <= phi(3) <= +inf
    -inf <= phi(4) <= +inf
    -inf <= phi(5) <= +inf
    -inf <= phi(6) <= +inf
    -inf <= phi(7) <= +inf
    -inf <= phi(8) <= +inf
    -inf <= phi(9) <= +inf
    -inf <= phi(10) <= +inf
    -inf <= phi(11) <= +inf
    -inf <= phi(12) <= +inf
    -inf <= phi(13) <= +inf
    -inf <= phi(14) <= +inf
    -inf <= phi(15) <= +inf
    -inf <= phi(16) <= +inf
    -inf <= phi(17) <= +inf
    -inf <= phi(18) <= +inf
    -inf <= phi(19) <= +inf
    -inf <= phi(20) <= +inf
    -inf <= phi(21) <= +inf
    -inf <= phi(22) <= +inf
    -inf <= phi(23) <= +inf
    -inf <= phi(24) <= +inf
end
