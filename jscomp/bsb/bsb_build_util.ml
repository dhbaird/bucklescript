(* Copyright (C) 2015-2016 Bloomberg Finance L.P.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * In addition to the permissions granted to you by the LGPL, you may combine
 * or link a "work that uses the Library" with a publicly distributed version
 * of this file to produce a combined library or application, then distribute
 * that combined work under the terms of your choosing, with no requirement
 * to comply with the obligations normally placed on you by section 4 of the
 * LGPL version 3 (or the corresponding section of a later version of the LGPL
 * should you choose to use a later version).
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA. *)

let flag_concat flag xs = 
  xs 
  |> Ext_list.flat_map (fun x -> [flag ; x])
  |> String.concat " "
let (//) = Ext_filename.combine

(* we use lazy $src_root_dir *)

(* assume build dir is fixed to be _build *)
let rel_dir = Filename.parent_dir_name 


let convert_path = 
  if Sys.unix then fun x -> x 
  else if Sys.win32 || Sys.cygwin then 
    Ext_filename.replace_slash_backward 
  else failwith ("Unknown OS : " ^ Sys.os_type)

let convert_and_resolve_path = 
  if Sys.unix then Bsb_config.proj_rel  
  else 
  if Sys.win32 || Sys.cygwin then 
    fun (p:string) -> 
      let p = Ext_filename.replace_slash_backward p in
      Bsb_config.proj_rel p 
  else failwith ("Unknown OS :" ^ Sys.os_type)
(* we only need convert the path in the begining*)

(** converting a file from Linux path format to Windows *)
(*
let convert_and_resolve_file = 
  if Sys.unix then fun (p : string) -> 
    if Ext_filename.no_slash p 0 (String.length p) then p 
    else Bsb_config.proj_rel  p 
  else 
  if Sys.win32 || Sys.cygwin then 
    fun (p:string) -> 
      let p1 = Ext_filename.replace_slash_backward p in
      if p1 == p then 
        p 
      else 
       Bsb_config.proj_rel p1 
  else failwith ("Unknown OS :" ^ Sys.os_type)
*)
(**
   if [Sys.executable_name] gives an absolute path, 
   nothing needs to be done
   if it is a relative path 

   there are two cases: 
   - bsb.exe
   - ./bsb.exe 
   The first should also not be touched
   Only the latter need be adapted based on project root  
*)
let get_bsc_bsdep cwd = 
  let dir = 
    Filename.dirname (Ext_filename.normalize_absolute_path (cwd // Sys.executable_name))in 
  dir // "bsc.exe", dir // "bsb_helper.exe"

(** 
{[
mkp "a/b/c/d"
]}
*)
let rec mkp dir = 
  if not (Sys.file_exists dir) then 
    let parent_dir  = Filename.dirname dir in
    if  parent_dir = Filename.current_dir_name then 
      Unix.mkdir dir 0o777 (* leaf node *)
    else 
      begin 
        mkp parent_dir ; 
        Unix.mkdir dir 0o777 
      end
  else if not  @@ Sys.is_directory dir then 
    failwith ( dir ^ " exists but it is not a directory, plz remove it first")
  else ()


let get_list_string s = 
  Ext_array.to_list_map (fun (x : Bsb_json.t) ->
      match x with 
      | `Str x -> Some x.str
      | _ -> None
    ) s   


