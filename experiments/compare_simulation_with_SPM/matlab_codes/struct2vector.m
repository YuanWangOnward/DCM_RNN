function vector = struct2vector(struct_in)
    
    names = sort(fieldnames(struct_in));
    
    vector_length = length(names);
    vector = zeros(vector_length, 1);
    for i = 1: length(names)
        vector(i) = struct_in.(names{i});
    end
end