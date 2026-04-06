module adder_tb;

	reg [3:0] A,B;
	wire [4:0] Sum;

	Adder4bit adder_dut(.A(A), .B(B), .Sum(Sum));

	initial begin
		$monitor("a=%d b=%d sum=%d", A, B, Sum);
		A = 0; B= 0; 		
		#10; A = 4; B = 3; 
		#10; A = 7; B = 8; 
		#10; A = 15; B = 1; 
		#100;

        $finish;
	end
	endmodule
	
	
	
	

    

  
    