function output = forward(input)
    global l1_weight l1_bias l2_weight l2_bias l3_weight l3_bias l4_weight l4_bias ...
        l5_weight l5_bias l6_weight l6_bias de1_weight de1_bias de2_weight de2_bias 
    y1=relu(input*l1_weight+l1_bias');
    y2=relu(y1*l2_weight+l2_bias');
    y3=relu(y2*l3_weight+l3_bias');
    y4=relu(y3*l4_weight+l4_bias');
    y5=relu(y4*l5_weight+l5_bias');
    y6=(y5*l6_weight+l6_bias');
    y8=relu(y6*de1_weight+de1_bias');
    output=y8*de2_weight+de2_bias';
end
function output = relu(input)
    output=max(input,0);
end