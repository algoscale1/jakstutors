$('.input-file').change(function(e){
    $('.input-model').val(e.target.files[0].name);
    console.log($('.input-model').val(e.target.files[0].name));
    $('.file-selector').submit();
   // $('.input-model').val('');
});

var status = false;

$('#myModal').blur(function(e){
    e.preventDefault()
})

//var myApp = angular.module('myApp',['ui.router']);
//
//myApp.controller('baseController',['$scope',function($scope){
//    $scope.title = "testing";
//}]);
//
//myApp.config(function($stateProvider, $urlRouterProvider,$locationProvider) {
//    $locationProvider.html5Mode(true);
//    $urlRouterProvider.otherwise('/index');
//    $stateProvider
//        .state('index',{
//            url : '/index',
//            templateUrl : '/templates/home.html'
//        })
//        .state('login',{
//            url : '/login',
//            templateUrl : '/templates/login.html'
//        })
//        .state('signup',{
//            url : '/signup',
//            templateUrl : '/templates/signup.html'
//        })
//});
