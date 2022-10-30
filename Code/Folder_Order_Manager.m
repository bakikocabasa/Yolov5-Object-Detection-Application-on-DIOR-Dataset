
fullFileName = fullfile(pwd, 'train.txt')
opts = detectImportOptions('train.txt');
opts = setvartype(opts, 'string');  %or 'char' if you prefer
train = readtable(fullFileName,opts);


fullFileName2 = fullfile(pwd, 'test.txt')
opts2 = detectImportOptions('test.txt');
opts2 = setvartype(opts2, 'string');  %or 'char' if you prefer
test = readtable(fullFileName2,opts2);

fullFileName3= fullfile(pwd, 'val.txt')
opts3 = detectImportOptions('val.txt');
opts3 = setvartype(opts3, 'string');  %or 'char' if you prefer
val = readtable(fullFileName3,opts3);

source_train=strcat(pwd,'\images\train\',train.Var1(:,:),'.jpg')

source_val=strcat(pwd,'\images\train\',val.Var1(:,:),'.jpg')

source_test=strcat(pwd,'\images\test\',test.Var1(:,:),'.jpg')

dest_train=strcat(pwd,'\images\train_update\',train.Var1(:,:),'.jpg')

dest_val=strcat(pwd,'\images\val_update\',val.Var1(:,:),'.jpg')

dest_test=strcat(pwd,'\images\test_update\',test.Var1(:,:),'.jpg')



source_train_cell = convertStringsToChars(source_train)
dest_train_cell = convertStringsToChars(dest_train)

source_val_cell = convertStringsToChars(source_val)
dest_val_cell = convertStringsToChars(dest_val)

source_test_cell = convertStringsToChars(source_test)
dest_test_cell = convertStringsToChars(dest_test)

for i=2:length(source_train_cell)
movefile(source_train_cell{i,:},dest_train_cell{i,:})   
end

for i=1:length(source_test_cell)
movefile(source_test_cell{i,:},dest_test_cell{i,:})   
end

for i=1:length(source_val_cell)
movefile(source_val_cell{i,:},dest_val_cell{i,:})   
end
