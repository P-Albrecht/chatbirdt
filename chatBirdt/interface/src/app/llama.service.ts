import { Injectable } from '@angular/core';

import { HttpClient, HttpParams, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { map, catchError} from 'rxjs/operators';


@Injectable()
export class LlamaService {

  //baseURL: string = "http://chat-birdt-backend:5000/";
  //baseURL: string = "http://172.21.0.6:5000/";
  baseURL: string = "http://localhost:5000/";
  //baseURL: string = "backend/";

  constructor(private http: HttpClient) {
  }

  test(): Observable<any> {
    //return this.http.get(this.baseURL + 'question/' + text);
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
    });
    return this.http.get(this.baseURL + "test", { headers });
  }

  question(text: string): Observable<any> {
    //return this.http.get(this.baseURL + 'question/' + text);
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
    });
    return this.http.get(this.baseURL + "question/" + text, { headers });
  }

  questionDocument(text: string): Observable<any> {
    //return this.http.get(this.baseURL + 'question/' + text);
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
    });
    return this.http.get(this.baseURL + "questionDocument/" + text, { headers });
  }

  chatDocument(text: string, texts: string[], fromAI: string[]): Observable<any> {
    //return this.http.get(this.baseURL + 'question/' + text);
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
    });

    let params = new HttpParams();

    texts.forEach((element, i) => {
      params = params.append('texts[]', element);
      params = params.append('fromAI[]', fromAI[i]);
    })

    return this.http.get(this.baseURL + "chatDocument/" + text, { headers:headers, params: params });
  }
}
