$('.input-file').change(function(e){
    $('.input-model').val(e.target.files[0].name);
    $('.file-selector').submit();
    $('.input-model').val('');
});

//var myApp = angular.module('myApp',['ui.router']);

//myApp.controller('baseController',['$scope',function($scope){

//}]);

//myApp.config(function($stateProvider, $urlRouterProvider) {
//    $urlRouterProvider.otherwise('/home');
//    $stateProvider
//        .state('home',{
//            url : '/home',
//            templateUrl : '/templates/components/home.html'
//        })
//        .state('signIn',{
//            url : '/signIn',
//            templateUrl : '/templates/components/signIn.html'
//        })
//});