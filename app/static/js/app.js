$('.input-file').change(function(e){
    $('.input-model').val(e.target.files[0].name);
    $('.file-selector').submit();
	console.log(e.target.files)
});
