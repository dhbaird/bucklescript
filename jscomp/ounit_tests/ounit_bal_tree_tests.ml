let ((>::),
     (>:::)) = OUnit.((>::),(>:::))

let (=~) = OUnit.assert_equal

let suites = 
    __FILE__ >:::
    [
        __LOC__ >:: begin fun _ ->
            OUnit.assert_bool __LOC__
             (Bal_tree.invariant 
                (Bal_tree.of_array (Array.init 1000 (fun n -> n))))
        end;
        __LOC__ >:: begin fun _ ->
            OUnit.assert_bool __LOC__
             (Bal_tree.invariant 
                (Bal_tree.of_array (Array.init 1000 (fun n -> 1000-n))))
        end;
        __LOC__ >:: begin fun _ ->
            OUnit.assert_bool __LOC__
             (Bal_tree.invariant 
                (Bal_tree.of_array (Array.init 1000 (fun n -> Random.int 1000))))
        end;
        
    ]

